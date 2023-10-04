import cv2
import math
import numpy as np

class Connected_component:
    """Class to strore connected component track"""
    
    _AREA_INCREASE_COEF = 0.1

    def __init__(self, raw_img, binary_img, index, stats, points, swt=False):
        self.index = index

        self.swt = swt

        self.width = stats[2]
        self.height = stats[3]
        self.area = stats[4]

        self.left = np.max((stats[0] - int(self.width * 0.05), 0))
        self.top = np.max((stats[1] - int(self.height * 0.05), 0))
        self.width = int(self.width * 1.1)
        self.height = int(self.height * 1.1)

        self.right = np.min([self.left + self.width, raw_img.shape[1]])
        self.bottom = np.min([self.top + self.height, raw_img.shape[0]])

        self.width = self.right - self.left
        self.height = self.bottom - self.top        

        self.points = points

        self.image_name = ''

        # Mask raw image by connected component points
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        points = np.array(self.points)
        mask[points[:,1] - self.top, points[:,0] - self.left] = 1

        self.raw_img = raw_img[self.top:self.bottom, self.left:self.right]
        self.fit_image_to_full_range()
        self.raw_img = cv2.bitwise_and(self.raw_img, self.raw_img, mask=mask)

        self.binary_img = binary_img[self.top:self.bottom, self.left:self.right]

        self.correlation_max = 0
        self.shift = 0

        self._center_line = None
        self._center_line_left = None
        self._center_line_right = None
        self._visualized_img = None


    @classmethod
    def filter_components(cls, components, stats, raw_image, binary_image, labels, labels_sparse, area_threshold_min=50, area_threshold_max=5*10**5, swt=False):
        filtered_components = []
        k = 0
        for i in range(1, components):
            if stats[i][4] >= area_threshold_min and stats[i][4] <= area_threshold_max:
                coords = np.unravel_index(labels_sparse[i].data, labels.shape)
                points = tuple(zip(coords[1], coords[0]))
                filtered_components.append(Connected_component(raw_image, binary_image, k, stats[i], points, swt))
                k += 1
        return filtered_components

    @property
    def visualized_img(self):
        if self._visualized_img is None:
            self._visualized_img = self.get_visualized_img()
        return self._visualized_img

    @property
    def center_line_left(self):
        if self._center_line_left is None:
            self.get_center_line()
        return self._center_line_left

    @property
    def center_line_right(self):
        if self._center_line_right is None:
            self.get_center_line()
        return self._center_line_right

    @property
    def center_line(self):
        if self._center_line is None:
            self.get_center_line(self.swt)
        return self._center_line

    def draw_component(self, img):
        for i in range(self.binary_img.shape[0]):
            for j in range(self.binary_img.shape[1]):
                #if self.binary_img[i, j] == 255:
                img[i + self.top, j + self.left] = [0, 0, self.binary_img[i, j]]
        return img



    def get_center_line_with_SWT(self, diagnostics=False, show_results=False):
        img = self.raw_img
        thres, img2 = cv2.threshold(img, 7, 255, cv2.THRESH_BINARY)
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        if show_results:
            cv2.imshow('', img2)
            cv2.waitKey()

        edges = cv2.Canny(img2, 150, 400, apertureSize=3) #阈值1，2，  apertureSize就是Sobel算子的大小

        # Create gradient map using Sobel
        #Sobel(src, ddepth, dx, dy[, ksize[, scale[, delta[, borderType]]]])利用Sobel算子进行图像梯度计算；
        # src：输入图像；ddepth: 输出图像的深度（可以理解为数据类型），-1表示与原图像相同的深度；
        # dx,dy:当组合为dx=1,dy=0时求x方向的一阶导数，当组合为dx=0,dy=1时求y方向的一阶导数（如果同时为1，通常得不到想要的结果）
        #ksize:（可选参数）Sobel算子的大小，必须是1,3,5或者7,默认为3
        sobelx64f = cv2.Sobel(img,cv2.CV_64F, 1, 0, ksize=-1)
        sobely64f = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=-1)

        theta = np.arctan2(sobely64f, sobelx64f)

        if diagnostics:
            #cv2.imwrite('edges.jpg',edges) #保存图像
            cv2.imwrite('edges.bmp',edges)
            cv2.imwrite('sobelx64f.bmp', np.absolute(sobelx64f))  #对数组中的每一个元素求其绝对值
            cv2.imwrite('sobely64f.bmp', np.absolute(sobely64f))
            # amplify theta for visual inspection
            theta_visible = (theta + np.pi)*255/(2*np.pi)
            cv2.imwrite('theta.bmp', theta_visible)
        
        if show_results:
            cv2.imshow('', edges)
            cv2.waitKey()


        # create empty image, initialized to infinity
        swt = np.empty(theta.shape)
        swt[:] = np.Infinity  #表示+∞，是没有确切的数值的,类型为浮点型
        rays = []

        # print(time.perf_counter() - t0)

        # now iterate over pixels in image, checking Canny to see if we're on an edge.
        # if we are, follow a normal a ray to either the next edge or image border
        # edgesSparse = scipy.sparse.coo_matrix(edges)
        step_x_g = sobelx64f
        step_y_g = sobely64f
        mag_g = np.sqrt(step_x_g * step_x_g + step_y_g * step_y_g )
        
        np.seterr(divide='ignore',invalid='ignore')
        grad_x_g = step_x_g / mag_g
        grad_y_g = step_y_g / mag_g

        edges_indicies = np.argwhere(edges)

        for (y, x) in edges_indicies:
            # step_x = step_x_g[y, x]
            # step_y = step_y_g[y, x]
            # mag = mag_g[y, x]
            grad_x = grad_x_g[y, x]
            grad_y = grad_y_g[y, x]

            if np.isnan(grad_x) or np.isnan(grad_y):
                continue

            ray = []  #射线
            ray.append((x, y))
            prev_x, prev_y, i = x, y, 0         #射线方向p点的坐标值
            while True:
                i += 1
                cur_x = math.floor(x + grad_x * i) #函数总是返回小于等于一个给定数字的最大整数
                cur_y = math.floor(y + grad_y * i)

                # we have moved to the next pixel!
                if cur_x != prev_x or cur_y != prev_y:  #不等于
                    # Если вышли за границу изображения
                    if (not (0 <= cur_x < edges.shape[1])) or (not(0 <= cur_y < edges.shape[0])):
                        break

                    # Попали в границу
                    if edges[cur_y, cur_x] > 0:                        
                        ray.append((cur_x, cur_y))
                        # theta_point = theta[y, x]
                        # alpha = theta[cur_y, cur_x]
                        beta = np.max((-1.0, grad_x * -grad_x_g[cur_y, cur_x] + grad_y * -grad_y_g[cur_y, cur_x]))
                        beta = np.min((1.0, beta))
                        if math.acos(beta) < np.pi/2.0:  #反余弦值
                            thickness = math.sqrt( (cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y) )
                            for (rp_x, rp_y) in ray:
                                swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                            rays.append(ray)
                        break
                    # this is positioned at end to ensure we don't add a point beyond image boundary
                    ray.append((cur_x, cur_y))

                    prev_x = cur_x
                    prev_y = cur_y

        # Compute median SWT
        for ray in rays:
            median = np.median([swt[y, x] for (x, y) in ray])
            for (x, y) in ray:
                swt[y, x] = min(median, swt[y, x])

        if show_results:
            cv2.imshow('', swt * 100)
            cv2.waitKey()
        if diagnostics:
            cv2.imwrite('swt.bmp', swt * 100)

        center_line = []
        lengths = []
        
        for ray in rays:
            i_x = 0
            i_y = 0
            i_sum = 0
            for (x, y) in ray:
                i_x = i_x + img[y, x] * x
                i_y = i_y + img[y, x] * y
                i_sum = i_sum + img[y, x]
            center_line.append((i_x / i_sum, i_y / i_sum))
            lengths.append(np.sqrt((ray[0][0]-ray[-1][0])**2 + (ray[0][1]-ray[-1][1])**2))

        # Индекс вершины параболы (наивысшая точка)
        min_ind = np.argmin(np.array(center_line)[:,1])
        self.top_x, self.top_y = center_line[min_ind]

        sorted_center_line_right = sorted(filter(lambda x: x[0] < center_line[min_ind][0], center_line), key=lambda x: math.sqrt((x[0] - center_line[min_ind][0])**2 + (x[1] - center_line[min_ind][1])**2))
        sorted_center_line_left = sorted(filter(lambda x: x[0] > center_line[min_ind][0], center_line), key=lambda x: math.sqrt((x[0] - center_line[min_ind][0])**2 + (x[1] - center_line[min_ind][1])**2))

        self._center_line = sorted_center_line_right + [center_line[min_ind]] + sorted_center_line_left
        self._center_line_left = np.array(sorted_center_line_left)
        self._center_line_right = np.array(sorted_center_line_right)
        

    def get_center_line_original(self):

        center_line = []

        self.raw_img = cv2.GaussianBlur(self.raw_img,(3, 3),0)

        range_y = range(self.raw_img.shape[0])
        for x in range(self.raw_img.shape[1]):
            y_sum = np.sum(self.raw_img[:,x]*np.array(range_y))
            y_center = y_sum / np.sum(self.raw_img[:,x])
            if not np.isnan(y_center):
                center_line.append([y_center, x])

        center_line = np.array(center_line)
        self.top_y, self.top_x = center_line[np.argmin(center_line, 0)[0], :]

        center_line_left = []
        center_line_right = []

        top_x = int(round(self.top_x))
        top_y = int(round(self.top_y))

        range_x1 = range(top_x)
        range_x2 = range(top_x, self.raw_img.shape[1])
        for y in range(top_y + 1, self.raw_img.shape[0]):
            xx_sum1 = np.sum(self.raw_img[y,:top_x]*np.array(range_x1))
            xx_sum2 = np.sum(self.raw_img[y,top_x:]*np.array(range_x2))
            x_sum1 = np.sum(self.raw_img[y,:top_x])
            x_sum2 = np.sum(self.raw_img[y,top_x:])
            if x_sum1 > 0:
                x_center1 = xx_sum1 / x_sum1
            else:
                x_center1 = np.nan
            if x_sum2 > 0:
                x_center2 = xx_sum2 / x_sum2
            else:
                x_center2 = np.nan
            if not np.isnan(x_center1):
                center_line_left.append([x_center1, y])
            if not np.isnan(x_center2):
                center_line_right.append([x_center2, y])
        self._center_line = center_line_left + center_line_right
        self._center_line_left = np.array(center_line_left)
        self._center_line_right = np.array(center_line_right)
        return self._center_line

    def get_center_line(self, swt=False):
        if not swt:
            self.get_center_line_original()
        else:
            self.get_center_line_with_SWT()


    def get_visualized_img(self):
        visualized_img = cv2.cvtColor(self.raw_img, cv2.COLOR_GRAY2BGR)
        
        for point in self.center_line:
            try:
                # for (x, y) in self.center_line[::]:
                #     visualized_img=cv2.drawMarker(visualized_img, (int(x), int(y)), (255, 0, 0))
                visualized_img[int(point[1]), int(point[0])] = [255, 0, 0]
            except:
                pass
        
        # visualized_img = cv2.circle(visualized_img, (int(self.top_x), int(self.top_y)), 1, (0,0,255))
        return visualized_img

    def fit_image_to_full_range(self):
        img_max = max(self.raw_img.flatten())
        img_min = min(self.raw_img.flatten())
        self.raw_img = np.array((self.raw_img - img_min) / (img_max -img_min ) * 255, dtype=np.uint8)

    def point_belongs_num(self, points):
        points_num = len(points)
        points_belongs = 0
        points_dont_belongs = 0
        threshold_point_num = points_num * 0.2
        for point in points:
            if point in self.points:
                points_belongs += 1
                if points_belongs > threshold_point_num:
                    return True
            else:
                if points_dont_belongs > points_num * 0.2:
                    break
                points_dont_belongs += 1
        return False

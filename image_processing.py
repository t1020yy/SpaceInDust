import json
import glob
import pickle
import random
from typing import List, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

import modeling
from report import make_report
from excel_report import save_tracks_to_excel
from particle_track import Particle_track
from connected_component import Connected_component
from modeling_parameters import ModelingParabolaParameters
        

def findEllipses(img, min_a=30, max_a=1000, threshold=20):
    pixels = cv2.findNonZero(img)

    results = []

    itteration = 0
    max_itterations = len(pixels)**3 / 2

    for i in range(len(pixels[:,0,:])):
        pixel1 = pixels[i,0,:]
        for j in range(i, len(pixels[:,0,:])):
            pixel2 = pixels[j,0,:]
            a = np.linalg.norm(pixel1 - pixel2)/2 #((pixel1[0] - pixel2[0])**2 + (pixel1[1] - pixel2[1])**2)**0.5 / 2 
            if a > min_a and a < max_a:
                accumulator = np.zeros(2000, dtype=np.int32)
                x0 = (pixel1[0] - pixel2[0]) / 2
                y0 = (pixel1[1] - pixel2[1]) / 2
                alfa = np.arctan((pixel2[1] - pixel1[1])/(pixel2[0] - pixel1[0]))
                for pixel3 in pixels[:,0,:]:
                    if not np.array_equal(pixel3, pixel1) and not np.array_equal(pixel3, pixel2):
                        d = np.linalg.norm(np.array([x0, y0], dtype=np.int32) - pixel3) #((x0 - pixel3[0])**2 + (y0 - pixel3[1])**2)**0.5
                        if d > min_a:
                            f = np.linalg.norm(pixel2 - pixel3) #((pixel2[0] - pixel3[0])**2 + (pixel2[1] - pixel3[1])**2)**0.5
                            tau = np.arccos((a**2 + d**2 - f**2) / (2*a*d))
                            b = ((a**2 * d**2 * np.sin(tau)**2)/(a**2 - d**2 * np.cos(tau)**2))**0.5
                            try:
                                accumulator[int(b)] += 1
                            except:
                                pass
                
                if max(accumulator) > threshold:
                    results.append([x0, y0, a, accumulator.argmax(), alfa])
                itteration += len(pixels)
                print(f'Itteration {itteration} from {max_itterations}')
    return results


def parabolaHough(img, threshold=20):
    pixels = cv2.findNonZero(img)

    results = []

    accumulator = np.zeros((200,200,200))

    for i in range(len(pixels[:,0,:])):
        pixel1 = pixels[i,0,:]
        for a in range(200):
            for b in range(200):
                c = pixel1[1] - a*pixel1[0]**2 - b*pixel1[0]**2
                if c > 0 and c < 200:
                    accumulator[a,b,int(c)] += 1
                
    if max(accumulator) > threshold:
        pass

    return results


def parabola_from_contour(contour):

    accumulator = []
    contour = np.array(contour)[:,0,:]

    y_max_index = contour[:,1].argmin(axis=0)
    x_a = contour[y_max_index, 0]
    y_a = contour[y_max_index, 1]

    for point in contour:
        a = (point[1] - y_a) / (point[0] - x_a)**2
        if not np.isnan(a) and not np.isinf(a):
            accumulator.append(a)

    return x_a, y_a, accumulator


def draw_parabola(img, parabola):

    points = []
    x_a = parabola[0]
    y_a = parabola[1]
    a = np.average(parabola[2])
    
    for y in range(y_a, 1200):
        x1 = ((y - y_a) / a)**0.5 + x_a
        x2 = - ((y - y_a) / a)**0.5 + x_a
        if not np.isnan(x1) and not np.isnan(x2) and not np.isinf(x1) and not np.isinf(x1):
            points.append([y, int(x1)])
            points.append([y, int(x2)])
        
    for point in points:
        img = cv2.circle(img, (point[1], point[0]), 5, (255,0,0))

    return img


def rectifyImages(img1, img2, cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T):
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, img1.shape[::-1], R, T, alpha=0, flags=0)

    mapx1, mapy1 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, img1.shape[::-1], cv2.CV_32F)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, img1.shape[::-1], cv2.CV_32F)

    #width = max(roi1[2], roi2[2])
    #height = max(roi1[3], roi2[3])
    
    img_rect1 = cv2.remap(img1, mapx1, mapy1, cv2.INTER_LINEAR)#[roi1[1]:roi1[1]+height, roi1[0]:roi1[0]+width]
    img_rect2 = cv2.remap(img2, mapx2, mapy2, cv2.INTER_LINEAR)#[roi2[1]:roi2[1]+height, roi2[0]:roi2[0]+width]

    #total_size = (max(img_rect1.shape[0], img_rect2.shape[0]), img_rect1.shape[1] + img_rect2.shape[1])
    #img = np.zeros(total_size, dtype=np.uint8)
    #img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
    #img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2

    return img_rect1, img_rect2, P1, P2


def drawRectifiedImages(img1, img2):
    
    img = np.hstack((img1, img2))

    for i in range(20, img.shape[0], 25):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
            
    cv2.namedWindow('imgRectified', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('imgRectified', 1000, 600)
    cv2.imshow('imgRectified', img)            
    
    cv2.waitKey()

    return


def draw_tracks(tracks):

    if len(tracks) > 0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mins = []
        maxs = []

        for track in tracks:
            points = track.flatten_3d_points
        
            ax.scatter(points[:,0,0], points[:,0,1], points[:,0,2], marker='o')
            mins.append(min(points[:,0,2]))
            maxs.append(max(points[:,0,2]))
            
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

        ax.set_zlim(min(mins) - 2, max(maxs) + 2)

        ax.set_xlabel('X, mm')
        ax.set_ylabel('Y, mm')
        ax.set_zlabel('Z, mm')

        ax.view_init(-70, -90)
            
        plt.show()


def get_angle_between_plane_and_camera(A, B, C):
    # 计算粒子运动平面的法向量 n1

    n1 = np.array([A, B, 1], dtype=float)
    n1 /= np.linalg.norm(n1)

    # 提取相机的旋转矩阵 R 的第三列作为相机观察面的法向量 n2
    cam1 = {
        'R': np.array([[1., 0., 0.],
                       [0, 1., 0.],
                       [0., 0., 1.]])
    }
    n2 = cam1['R'][:, 2]

    # 计算夹角
    angle = np.arccos(np.dot(n1, n2))

    return np.degrees(angle)

# def get_angle_between_plane_and_camera(A, B, C):
#     cam1 = {
#     'R': np.array([[1., 0., 0.],
#                    [0, 1., 0.],
#                    [0., 0., 1.]]),
#     'T': np.array([0., 0., 0.]),
#     'K': np.array([[12900, 0, 960], [0, 12900, 600], [0, 0, 1]], dtype=float),
#     'dist': np.array([0, 0, 0, 0], dtype=float)
#     }

#     # 获取相机的旋转矩阵和平移向量
#     R = cam1['R']
#     T = cam1['T']

#     # 光轴向量表示为 [0, 0, 1]
#     optical_axis = np.array([0, 0, 1])

#     # 将光轴向量转换为相机坐标系下的向量
#     camera_coords = np.linalg.inv(cam1['K']) @ optical_axis

#     # 计算相机的位置
#     camera_position = -R.T @ T


#     # 计算相机到粒子所在平面的距离
#     distance_to_plane = abs(A * camera_position[0] + B * camera_position[1] + C)

#     # 计算相机与粒子所在平面的夹角
#     angle = np.arctan(distance_to_plane / np.linalg.norm(camera_coords))

#     # 将夹角转换为角度
#     # angle_degrees = np.degrees(angle)
#     # print(angle_degrees)
#     return np.degrees(angle)

def process_simulation_results(simulation_results: List[Tuple[ModelingParabolaParameters, Particle_track]]):

    angle_dif = []
    velocity_dif = []
    teta = []
    z = []
    AA = []
    BB = []
    CC = []
    FF = []
    TT = []
    for params, particle in simulation_results:
        angle_dif.append(params.start_angle / np.pi * 180 - particle.Alpha)
        velocity_dif.append(params.start_speed - particle.V0)
        # teta.append(get_angle_between_plane_and_camera(params.plane_parameter_A, params.plane_parameter_B, params.plane_parameter_C))
        teta_value = get_angle_between_plane_and_camera(params.plane_parameter_A, params.plane_parameter_B, params.plane_parameter_C)
        teta.append(teta_value)
        AA.append(params.plane_parameter_C)
        BB.append(params.start_angle/ np.pi * 180)
        CC.append(particle.Alpha)
        z.append(params.plane_parameter_A)
        FF.append(params.F)  # 记录F的值
        TT.append(params.theta)
    
    print(f'Всего смоделировано {len(teta)} треков.')
    

    # plt.plot(TT, angle_dif, 'bo')
    plt.plot(FF, TT, 'bo')
    plt.show()
    

if __name__ == "__main__":

    SIMULATION = True
    SIMULATION_CLAIBRATION_FILENAME = "./2022-10-24/calib 24 oct 2022 2.json"
    
    simulation_parameters : ModelingParabolaParameters
    simulation_results = []


    if (not SIMULATION):
        with open('experiments.json', 'r', encoding='utf8') as f:
            experiments = json.load(f)

        experiment = experiments[0]

        camera1_path = experiment['camera1_path']
        camera2_path = experiment['camera2_path']

        images_for_camera1 = sorted(glob.glob(camera1_path + experiment['image_file_mask']))

        assert len(images_for_camera1) > 0, f'Image files for processing not found!\nPath "{camera1_path}" is set in experiments.json file.'

        image_to_substruct1 = cv2.imread(images_for_camera1[experiment['background_image_index']])
        img_to_sub1 = cv2.cvtColor(image_to_substruct1, cv2.COLOR_BGR2GRAY)

        images_for_camera2 = sorted(glob.glob(camera2_path + experiment['image_file_mask']))
        image_to_substruct2 = cv2.imread(images_for_camera2[experiment['background_image_index']])
        img_to_sub2 = cv2.cvtColor(image_to_substruct2, cv2.COLOR_BGR2GRAY)
    else:
        img_to_sub1 = np.zeros((1200, 1920), dtype=np.uint8)
        img_to_sub2 = np.zeros((1200, 1920), dtype=np.uint8)

    try:
        if (not SIMULATION):
            path_to_calibration_file = experiment['path_to_calibration_data']
        else:
            path_to_calibration_file = SIMULATION_CLAIBRATION_FILENAME

        with open(path_to_calibration_file, 'r') as fp:
            calibration_data = json.load(fp)
            cameraMatrix1 = np.array(calibration_data['Camera1']['CameraMatrix'])
            distCoeffs1 = np.array(calibration_data['Camera1']['DistorsionCoefficients'])
            cameraMatrix2 = np.array(calibration_data['Camera2']['CameraMatrix'])
            distCoeffs2 = np.array(calibration_data['Camera2']['DistorsionCoefficients'])
            R = np.array(calibration_data['R'])
            T = np.array(calibration_data['T'])
            print('Calibration data is load from calibration_data.json') 
    except:
        print(f'Calibration data is not founded by path "{path_to_calibration_file}"') 

    tracks = []

    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img1', 700, 600)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img2', 700, 600)
    cv2.namedWindow('img3', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img3', 300, 300)
    cv2.namedWindow('img4', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img4', 300, 300)

    do_processing = True
    current_position = 0
    position_changed = True
    current_id = 0
    threshold= [5, 5]
    draw_graphics = True
    do_morph = False
    search_area_range = 500

    if (not SIMULATION):
        threshold[0] = experiment.get('initial_threshold', 5)
        threshold[1] = experiment.get('initial_threshold', 5)
        search_area_range = experiment.get('search_area_range', 300)

    while True:

        if position_changed or SIMULATION:
            
            if (not SIMULATION):
                if current_position == len(images_for_camera1) - 1:
                    current_position = 0
                fname1, fname2 = images_for_camera1[current_position], images_for_camera2[current_position]

                img1 = cv2.imread(fname1)
                img2 = cv2.imread(fname2)
            else:
                # # simulation_parameters.x_start_trajectory = -35 + 40 * random.random()
                # # simulation_parameters.y_start_trajectory = 20 * random.random()
                # simulation_parameters = ModelingParabolaParameters()
                # print(simulation_parameters.start_angle/np.pi*180)
                # simulation_parameters.start_angle = 72.98 / 180 * np.pi
                # print(simulation_parameters.start_angle/np.pi*180)
                # simulation_parameters.start_speed = -0.2905 * 10**3
                # simulation_parameters.x_start_trajectory = -0.128159
                # simulation_parameters.y_start_trajectory = 0
                # simulation_parameters.plane_parameter_A = -0.2659
                # simulation_parameters.plane_parameter_B = -0.007062
                # simulation_parameters.plane_parameter_C = 420.48045
                
                # simulation_parameters.particle_diameter = 0.032

                # img1, img2 = modeling.get_simulated_image(simulation_parameters, H=1200, W=1920)
                # print(simulation_parameters.x_start_trajectory)

                simulation_parameters = ModelingParabolaParameters()

                
                simulation_parameters.x_start_trajectory = -5 + 1 * random.random()
                simulation_parameters.y_start_trajectory = 10 + 1 * random.random()
                simulation_parameters.F = random.uniform(-10, -110)
                simulation_parameters.theta = random.uniform(0, 0.5)
                # simulation_parameters.x_start_trajectory = -0.128159
                # simulation_parameters.y_start_trajectory = 0
                simulation_parameters.start_angle = random.uniform(60 / 180 * np.pi,80 / 180 * np.pi)
                # simulation_parameters.start_angle = 72.98 / 180 * np.pi
                simulation_parameters.start_speed = random.uniform(-0.8 * 10**3, -0.5 * 10**3)
                simulation_parameters.plane_parameter_C = random.uniform(300, 450)
                simulation_parameters.plane_parameter_A = random.uniform(-1, 0)
                simulation_parameters.plane_parameter_B = random.uniform(-1, 0)
                # simulation_parameters.plane_parameter_A =  -0.266 
                # simulation_parameters.plane_parameter_B = -0.0071
                # simulation_parameters.plane_parameter_C = 420.48
                simulation_parameters.particle_diameter = 0.032
                                
                img1, img2 = modeling.get_simulated_image(simulation_parameters, H=1200, W=1920)

                if img1 is None:
                    print('Парабола вышла за границы изображения')
                    continue

                print("Start Angle:", simulation_parameters.start_angle / np.pi * 180)
                print("Start Speed:", simulation_parameters.start_speed)
                print("X Start Trajectory:", simulation_parameters.x_start_trajectory)
                print("Y Start Trajectory:", simulation_parameters.y_start_trajectory)
                print("plane_parameter_A:", simulation_parameters.plane_parameter_A)

        
            if len(img1.shape) > 2:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            else:
                gray1 = img1
            if len(img2.shape) > 2:
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            else:
                gray2 = img2
          
            # res1, res2, P1, P2 = rectifyImages(gray1, gray2, cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T)

            con_components1, con_components2 = [], []

            if do_processing:
                res1 = cv2.subtract(gray1, img_to_sub1)
                # res1 = res1[:1040,150:]

                #mask1 = np.ones((res1.shape[0], res1.shape[1]), dtype=np.uint8)
                #mask1[:mask1.shape[0], :600] = 0
                #res1 = cv2.bitwise_and(res1, res1, mask=mask1)

                res2 = cv2.subtract(gray2, img_to_sub2)
                # res2 = res2[:1040,150:]

                #mask2 = np.ones((res2.shape[0], res2.shape[1]), dtype=np.uint8)
                #mask2[:mask2.shape[0], :300] = 0
                #res2 = cv2.bitwise_and(res2, res2, mask=mask2)
                
                # res1, res2, P1, P2 = rectifyImages(res1, res2, cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T)
                
                P1 = cameraMatrix1 @ np.hstack((np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 0, 0]]).T))
                P2 = cameraMatrix1 @ np.hstack((R, np.array([T]).T))

                retval1, dst1 = cv2.threshold(res1, threshold[0], 255, cv2.THRESH_BINARY)
                retval2, dst2 = cv2.threshold(res2, threshold[1], 255, cv2.THRESH_BINARY)

                if do_morph:
                    kernel = np.ones((3,3),np.uint8)
                    dst1 = cv2.morphologyEx(dst1, cv2.MORPH_OPEN, kernel)
                    dst2 = cv2.morphologyEx(dst2, cv2.MORPH_OPEN, kernel)

                nb_components1, labels1, stats1, centroids1 = cv2.connectedComponentsWithStats(dst1, connectivity=8)
                nb_components2, labels2, stats2, centroids2 = cv2.connectedComponentsWithStats(dst2, connectivity=8)

                cols1 = np.arange(labels1.size)
                labels1_sparse = csr_matrix((cols1, (labels1.ravel(), cols1)), shape=(labels1.max() + 1, labels1.size))
                cols2 = np.arange(labels2.size)
                labels2_sparse = csr_matrix((cols2, (labels2.ravel(), cols2)), shape=(labels2.max() + 1, labels2.size))

                con_components1 = Connected_component.filter_components(nb_components1, stats1, res1, dst1, labels1, labels1_sparse, swt=False)
                con_components2 = Connected_component.filter_components(nb_components2, stats2, res2, dst2, labels2, labels2_sparse, swt=False)
            else:
                dst1, dst2 = res1, res2

            position_changed = False
        
        color1 = cv2.cvtColor(dst1, cv2.COLOR_GRAY2BGR)
        color2 = cv2.cvtColor(dst2, cv2.COLOR_GRAY2BGR)


        #grid_points1 = np.array([[1460, 1140]], dtype=np.float)
        #grid_points2 = np.array([[1220, 1173]], dtype=np.float)

        projmtx1 = np.dot(cameraMatrix1, np.hstack((np.identity(3), np.zeros((3,1)))))
        projmtx2 = np.dot(cameraMatrix2, np.hstack((R, T[np.newaxis, :].T)))

        #points_4d = cv2.triangulatePoints(projmtx1, projmtx2, grid_points1.T, grid_points2.T)
        #points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

        if draw_graphics:
            for component in con_components1:
                color1 = component.draw_component(color1)
                point = (component.left, component.top)
                cv2.putText(color1, f'id={component.index}', point, cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0), 2)

            for component in con_components2:
                color2 = component.draw_component(color2)
                point = (component.left, component.top)
                cv2.putText(color2, f'id={component.index}', point, cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 0, 0), 2)
        
        if len(con_components2) > 0 and current_id < len(con_components2):

            cur_component = con_components2[current_id]

            if draw_graphics:
                cv2.rectangle(color2, (cur_component.left, cur_component.top),
                                    (cur_component.right, cur_component.bottom), (0,255,0), 3)

            contour_zoomed = cur_component.raw_img

            cv2.imshow('img4', cur_component.visualized_img)
            
            search_area_left, search_area_right = cur_component.left - search_area_range, cur_component.right + search_area_range
            if search_area_left < 0:
                search_area_left = 0
            if search_area_right > dst1.shape[1]:
                search_area_right = dst1.shape[1]
            search_area_top, search_area_bottom = cur_component.top - 50, cur_component.bottom + 50
            if search_area_top < 0:
                search_area_top = 0
            if search_area_bottom > dst1.shape[0]:
                search_area_bottom = dst1.shape[0]

            search_dst1 = res1[search_area_top:search_area_bottom, search_area_left:search_area_right]

            try:
                res = cv2.matchTemplate(search_dst1, contour_zoomed, cv2.TM_CCOEFF_NORMED)
            except:
                pass
            finally:
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                matched_component = None

                if max_val > 0.25:               
                    matched_dst = dst1[max_loc[1] + search_area_top:max_loc[1] + search_area_top + cur_component.height,
                                    max_loc[0] + search_area_left:max_loc[0] + search_area_left + cur_component.width]
                    components_num, labels, stats, _ = cv2.connectedComponentsWithStats(matched_dst, connectivity=8)

                    if components_num > 1:
                        matched_component_id = stats[1:,4].argmax() + 1
                        matched_component_points = [(x + max_loc[0] + search_area_left, y + max_loc[1] + search_area_top) for y, x in zip(np.where(labels == matched_component_id)[0], np.where(labels == matched_component_id)[1])]

                        for component in con_components1:
                            if component.point_belongs_num(matched_component_points):
                                matched_component = component
                                break
                    try:
                        if draw_graphics:
                            cv2.rectangle(color1, (matched_component.left, matched_component.top),
                                                (matched_component.right, matched_component.bottom), (0,255,0), 3)

                        cv2.imshow('img3', matched_component.visualized_img)

                        matched_component.correlation_max = max_val
                        cur_component.correlation_max = max_val
                        matched_component.shift = max_loc
                        cur_component.shift = max_loc
                    except:
                        pass
                else:
                    empty_img = np.zeros((100, 100), dtype=np.uint8)
                    cv2.imshow('img3', empty_img)
        
        if (not SIMULATION):
            filename1 = fname1.split("\\")[-1]
            cv2.putText(color1, f'file={filename1}', (10,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
            
            filename2 = fname2.split("\\")[-1]
            cv2.putText(color2, f'file={filename2}', (10,30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)            

        cv2.imshow('img1', color1)
        cv2.imshow('img2', color2)

        if (SIMULATION):
            waitTime = 500
        else:
            waitTime = 0

        key = cv2.waitKey(waitTime)

        if (SIMULATION):
            try:
                if matched_component is not None:
                    particle = Particle_track(matched_component, cur_component, P1, P2)
                    particle.particle_radius = 100e-06
                    particle.particle_density = 2200
                    particle.voltage = 2500
                    particle.grid_height = 10
                    particle.image_name = "Modeled image"

                    tracks.append(particle)
                    print(f'Сохранено {len(tracks)} треков')

                    simulation_results.append((simulation_parameters, particle))

                else:
                    print(f'Не удалось найти стереопару параболы')
                                           
            except:
                pass
        
        
        if key==27:    # Esc key to stop
            break
        elif key==-1:  # normally -1 returned,so don't print it
            continue

        elif key==44:
            current_position -= 1
            current_id = 0
            position_changed = True
        elif key==46:
            current_position += 1
            current_id = 0
            position_changed = True

        elif key==47:
            if current_id > 0:
                current_id -= 1
        elif key==39:
            if current_id < len(con_components2) - 1:
                current_id += 1
        elif key == 49: # 1
            if threshold[0] > 0:
                threshold[0] -= 1
                print(f'Threshold[0] = {threshold[0]}')
                position_changed = True
        elif key == 50: # 2
            if threshold[0] < 255:
                threshold[0] += 1
                print(f'Threshold[0] = {threshold[0]}')
                position_changed = True
        elif key == 51: # 3
            if threshold[1] > 0:
                threshold[1] -= 1
                print(f'Threshold[1] = {threshold[1]}')
                position_changed = True
        elif key == 52: # 4
            if threshold[1] < 255:
                threshold[1] += 1
                print(f'Threshold[1] = {threshold[1]}')
                position_changed = True
        elif key == 100: # d
            draw_graphics = not draw_graphics
            position_changed = True
        elif key == 109: # m
            do_morph = not do_morph
            position_changed = True
        elif key==32:   # Space
            tracks.append(Particle_track(matched_component, cur_component, P1, P2))
            tracks[-1].particle_radius = experiment['particles_radius']
            tracks[-1].particle_density = experiment['particles_density']
            tracks[-1].voltage = experiment['voltage']
            tracks[-1].grid_height = experiment['grid_height']
            tracks[-1].image_name = fname1
            print(f'Сохранено {len(tracks)} треков')
        elif key == 112: # p
            draw_tracks(tracks)
        elif key == 114: # r
            drawRectifiedImages(color1, color2)
        elif key == 116: # t
            make_report(tracks)
        elif key == 117: # u
            process_simulation_results(simulation_results)
        elif key == 118: # v
            save_tracks_to_excel(experiment, tracks)
        elif key == 96: # `
            do_processing = not do_processing
            position_changed = True
        elif key == 122: # z
            with open('results.dat', 'wb') as f:
                pickle.dump(tracks, f)
        elif key == 120: # x
            with open('results.dat', 'rb') as f:
                tracks = pickle.load(f)
                print(f'Загружено {len(tracks)} треков, последний обработанный файл {tracks[-1].image_name}')
        else:
            print(key) # else print its value




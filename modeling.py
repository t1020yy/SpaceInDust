import cv2
import numpy as np

from modeling_parameters import ModelingParabolaParameters

def check_parabola_parameters(parabola_height, parabola_width, branches_height_ratio):
    if parabola_height < 250 or parabola_width < 200 or (branches_height_ratio < 0.2 or branches_height_ratio > 0.85):
        return False
    else:
        return True

def rotate_around_y(x, z, angle_degrees):
    """
    Rotate a point around the Y axis by a given angle.
    """
    angle_radians = np.radians(angle_degrees)
    x_rotated = x * np.cos(angle_radians) + z * np.sin(angle_radians)
    z_rotated = -x * np.sin(angle_radians) + z * np.cos(angle_radians)
    return x_rotated, z_rotated

def calculate_particle_position(x0, y0, v0, alpha, t, g=9.81*10**3):
    '''
    Возвращает координаты частицы во времени в соответствии с заданными параметрами левитации
    x0, y0 - координаты начала траектории (взлета, отрыва)
    v0 - начальная скорость
    alpha - начальный угол взлета
    t - время для которого необходимо вернуть координаты    
    g - ускорение свободного падения

    Траектория движения частицы будет соответствовать параболе y = a*x**2 + b*x + c, где
    a = g / (2 * vx0**2)
    b = 1 / vx0 * (vy0 - x0 * g / vx0)
    c = y0 + x0 / vx0 * (x0 * g / (2 * vx0) - vy0)
    '''
    vx0 = v0 * np.cos(np.radians(alpha))
    vy0 = -v0 * np.sin(np.radians(alpha))
    x = x0 + vx0 * t
    y = y0 + vy0 * t + 0.5 * g * t ** 2
    z = 0
    return (x, y, z)

def get_a_b_c(v0, alpha, x0, y0, g = 9.81*10**3):
    a = g / (2 * (v0 * np.cos(np.radians(alpha))) ** 2)
    b = - np.tan(np.radians(alpha)) - x0 * g / ((v0 * np.cos(np.radians(alpha)))**2)
    c = y0 + g * x0 / (2 * (v0 * np.cos(np.radians(alpha))) **2) + x0 * np.tan(np.radians(alpha))
    return a, b ,c

def calculate_trajectory(parameters: ModelingParabolaParameters):
    '''
    Возвращает массив координат частицы в соответствии с заданными параметрами левитации
    parameters - параметры моделирования параболы    
    '''
    # Координаты начального положения частицы [мм]
    x0 = parameters.x_start_trajectory
    y0 = parameters.y_start_trajectory
    
    # Начальная скорость [мм/с]
    v0 = parameters.start_speed

    # Угол взлета
    alpha = parameters.start_angle

    # Параметры плоскости в которой происходит левитация
    A = parameters.plane_parameter_A
    B = parameters.plane_parameter_B
    C = parameters.plane_parameter_C

    # Шаг по времени для расчета координат траектории частицы [с]
    dt = parameters.interval_time

    t = 0
    max_time = 2 * v0 * np.sin(np.radians(alpha)) / (9.81*10**3)

    trajectory = []    
    intersection_start = max(0, parameters.expose_time_start)
    intersection_end = min(max_time, parameters.expose_time_start + parameters.expose_time)

    while t <= intersection_end:
        if t >= intersection_start:
            (x, y, z) = calculate_particle_position(x0, y0, v0, alpha, t)
            x, z = rotate_around_y(x, z, B)
            z = z + C
            trajectory.append((x, y, z))
        t += dt

    return np.array(trajectory)


def gaus(x, y, x0, y0, sigma_x, sigma_y, delta_x, delta_y):
    '''
    Возвращает значение соотвествующее функции Гаусса с заданными параметрами
    '''
    return np.exp(-((x - x0)**2 / (2.0*((sigma_x / delta_x)**2)) + (y - y0)**2 / (2.0*((sigma_y / delta_y)**2))))


def add_intensity_subpixel(img, d, x0_list, y0_list, delta_x, delta_y, old_variant=False):
    '''
    Функция отображает траекторию частицы на изображении

    img - изображение для отображения траектории частицы
    d - размер изображения частицы в мм
    x0_list - список горизонтальных координат точек траектории
    y0_list - список вертикальных координат точек траектории
    delta_x, delta_y - размеры пикселя в мм
    old_variant - старый (более медленный) способ расчета
    '''

    # Размер изображения
    img_h, img_w = img.shape[0], img.shape[1]

    # Сигма для функции Гаусса
    size = d / 2
    sigma_x = size / 3
    sigma_y = size / 3
    delta_x = 5*10**-3 # mm
    delta_y = 5*10**-3 # mm
    # Размер изображения частицы в пикселях
    rpx_x = int(np.ceil((d / delta_x) / 2))
    rpx_y = int(np.ceil((d / delta_y) / 2))

    # Шаг интегрирования по поверхности пикселя
    dx = 0.1 * 10**-3 # mm
    dy = 0.1 * 10**-3 # mm

    #zhushidiao 
    if old_variant:
        # # Координаты центра матрицы в мм
        x_c = - delta_x * img_w / 2
        y_c = - delta_y * img_h / 2

        # Координатная сетка в одном пикселе
        xx = np.linspace(x_c, x_c + delta_x, int(delta_x / dx))
        yy = np.linspace(y_c, y_c + delta_y, int(delta_y / dy))
        xxx, yyy = np.meshgrid(xx, yy)

        for x0, y0 in zip(x0_list, y0_list):
            # Округленные координаты текущей точки траектории
            x_round = int(np.round(x0))
            y_round = int(np.round(y0))

            for x in range(x_round - rpx_x, x_round + rpx_x):
                for y in range(y_round - rpx_y, y_round + rpx_y):
                    if not (0 < x < img_w) or not (0 < y < img_h):
                        continue
                    xxxx = xxx + x * delta_x
                    yyyy = yyy + y * delta_y

                    g = gaus(xxxx, yyyy, x0, y0, sigma_x, sigma_y, delta_x, delta_y)
                    img[y, x] += np.sum(g) * (delta_x * delta_y)

# #daozhe
    else:
        # Координатная сетка для всего изображения частицы
        xx = np.linspace(0, rpx_x, int(rpx_x * delta_x / dx))
        yy = np.linspace(0, rpx_y, int(rpx_y * delta_y / dy))
        xxx, yyy = np.meshgrid(xx, yy)

        # Изображение всей частицы с повышенным разрешением
        g = gaus(xxx, yyy, rpx_x/2, rpx_y/2, sigma_x, sigma_y, delta_x, delta_y)

        # Количество разбиений в пикселе
        stepsx_in_pxl = int(np.round(delta_x / dx))
        stepsy_in_pxl = int(np.round(delta_y / dy))

        for x0, y0 in zip(x0_list, y0_list):
            # Округленные координаты текущей точки траектории
            x_round = int(np.round(x0))
            y_round = int(np.round(y0))

            for x in range(0, rpx_x):
                for y in range(0, rpx_y):
                    x1 = x * stepsx_in_pxl
                    x2 = x1 + stepsx_in_pxl
                    y1 = y * stepsy_in_pxl
                    y2 = y1 + stepsy_in_pxl

                    #  Проверка на выход за границы изображения
                    try:
                        x_cur = x_round + x - rpx_x // 2
                        y_cur = y_round + y - rpx_y // 2
                        if x_cur >= 0 and  y_cur >=0:
                            img[y_round + y - rpx_y // 2, x_round + x - rpx_x // 2] += np.sum(g[y1:y2, x1:x2]) * (delta_x * delta_y)
                    except IndexError:
                        pass
   
    return img


def project_trajectory_3d(cam1, cam2, trajectory_3d):
    # project the 3D points onto the image planes of cam1 and cam2

    # R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(cam1['K'], cam1['dist'], cam2['K'], cam2['dist'], (2048,1536), cam2['R'], cam2['T'], alpha=0, flags=0)

    # k1, r1, t1, _, _, _, _ = cv2.decomposeProjectionMatrix(P1)
    # k2, r2, t2, _, _, _, _ = cv2.decomposeProjectionMatrix(P2)
    
    projected_points_cam1, _ = cv2.projectPoints(trajectory_3d, cam1['R'], cam1['T'], cam1['K'], cam1['dist'])
    projected_points_cam2, _ = cv2.projectPoints(trajectory_3d, cam2['R'], cam2['T'], cam2['K'], cam2['dist'])
    
    # convert the projected points to pixel coordinates
    projected_points_cam1 = cv2.convertPointsToHomogeneous(projected_points_cam1)
    projected_points_cam2 = cv2.convertPointsToHomogeneous(projected_points_cam2)
    
    return projected_points_cam1[:,0,:2], projected_points_cam2[:,0,:2]

def calculate_parabola_parameters(projected_points_cam1):
    # Общая высота параболы
    parabola_height = abs(np.max(projected_points_cam1[:, 1]) - np.min(projected_points_cam1[:, 1]))
    # Высота параболы в разных ветвях
    h_1 = abs(np.min(projected_points_cam1[:, 1]) - projected_points_cam1[-1, 1])
    h_2 = abs(np.min(projected_points_cam1[:, 1]) - projected_points_cam1[0, 1])
    # Коэффициент отношения высот ветвей параболы
    branches_height_ratio = min(h_1, h_2) / parabola_height
    # Ширина параболы
    parabola_width = abs(projected_points_cam1[0, 0] - projected_points_cam1[-1, 0])

    return parabola_height, parabola_width, branches_height_ratio

def get_simulated_image(parameters: ModelingParabolaParameters):
    
    trajectory = calculate_trajectory(parameters)
    trajectory_2d_x = trajectory[:, 0]
    trajectory_2d_y = trajectory[:, 1]
    trajectory_3d_z = trajectory[:, 2]    

    trajectory_3d = np.array(list(zip(trajectory_2d_x, trajectory_2d_y, trajectory_3d_z)))
    cam1 = {
        'R': parameters.cam1_R,
        'T': parameters.cam1_T,
        'K': parameters.cam1_K,
        'dist': parameters.cam1_dist
    }
    cam2 = {
        'R': parameters.cam2_R,
        'T': parameters.cam2_T,
        'K': parameters.cam2_K,
        'dist': parameters.cam2_dist
    }
    # Проецирование 3D точек на плоскость изображений для камер cam1 и cam2
    projected_points_cam1, projected_points_cam2 = project_trajectory_3d(cam1, cam2, trajectory_3d)

    # Рассчитываем параметры параболы на изображении

    parabola_height, parabola_width, branches_height_ratio = calculate_parabola_parameters(projected_points_cam1)  
    H = parameters.image_height
    W = parameters.image_width

    THRESHOLD = 0.25

    not_valid_points_1 = len(projected_points_cam1[projected_points_cam1[:,0] < 0]) + \
                        len(projected_points_cam1[projected_points_cam1[:,0] > W]) + \
                        len(projected_points_cam1[projected_points_cam1[:,1] < 0]) + \
                        len(projected_points_cam1[projected_points_cam1[:,1] > H])
                        
    not_valid_points_2 = len(projected_points_cam2[projected_points_cam2[:,0] < 0]) + \
                        len(projected_points_cam2[projected_points_cam2[:,0] > W]) + \
                        len(projected_points_cam2[projected_points_cam2[:,1] < 0]) + \
                        len(projected_points_cam2[projected_points_cam2[:,1] > H])
    
    if not_valid_points_1 / len(projected_points_cam1) > THRESHOLD or not_valid_points_2 / len(projected_points_cam2) > THRESHOLD:
        return None, None, None, None, None, None, None, None
    
    d = parameters.particle_diameter
    # Шаг интегрирования 
    delta_x = parameters.x_integration_step # mm
    delta_y = parameters.y_integration_step

    img1 = np.zeros((H, W), dtype=float)
    img2 = np.zeros((H, W), dtype=float)
    
    img1_1 = add_intensity_subpixel(img1, d, projected_points_cam1[:,0], projected_points_cam1[:,1], delta_x, delta_y)
    img2_1 = add_intensity_subpixel(img2, d, projected_points_cam2[:,0], projected_points_cam2[:,1], delta_x, delta_y)
    
    img1_1 = (img1_1 / np.max(img1_1) * 30).astype(np.uint8)
    img2_1 = (img2_1 / np.max(img2_1) * 30).astype(np.uint8)
    
    return img1_1, img2_1, projected_points_cam1, projected_points_cam2, trajectory_3d, parabola_height, parabola_width, branches_height_ratio




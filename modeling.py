import cv2
import numpy as np

from modeling_parameters import ModelingParabolaParameters


def calculate_particle_position(x0, y0, v0, alpha, t, A, B, C, g=9.81*10**3):
    '''
    Возвращает координаты частицы во времени в соответствии с заданными параметрами левитации
    x0, y0 - координаты начала траектории (взлета, отрыва)
    v0 - начальная скорость
    alpha - начальный угол взлета
    t - время для которого необходимо вернуть координаты
    A, B, C - параметры плоскости, в которой перемещается частица
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
    z = (A * x + B * y + C)
    
    return (x, y, z)


def calculate_trajectory(x0, y0, v0, alpha, A, B, C, dt, max_time=3):
    '''
    Возвращает массив координат частицы в соответствии с заданными параметрами левитации
    x0, y0 - координаты начала траектории (взлета, отрыва)
    v0 - начальная скорость
    alpha - начальный угол взлета
    A, B, C - параметры плоскости, в которой перемещается частица
    dt - шаг расчета траектории по времени
    max_time - время полета частицы
    '''
    t = 0
    trajectory = []

    while t < max_time:
        (x, y, z) = calculate_particle_position(x0, y0, v0, alpha, t, A, B, C)
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


def get_simulated_image(parameters: ModelingParabolaParameters):
    d = parameters.particle_diameter
    
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

    # Шаг интегрирования 
    delta_x = parameters.x_integration_step # mm
    delta_y = parameters.y_integration_step

    trajectory = calculate_trajectory(x0, y0, v0, alpha, A, B, C, dt = dt, max_time=0.1)

    trajectory_2d_x = trajectory[:, 0]
    trajectory_2d_y = trajectory[:, 1]
    trajectory_3d_z = trajectory[:, 2]

# assume we have the 3D trajectory stored in variable "trajectory_3d"
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
    # project the 3D points onto the image planes of cam1 and cam2
    projected_points_cam1, projected_points_cam2 = project_trajectory_3d(cam1, cam2, trajectory_3d)

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
        return None, None, None
   
    img1 = np.zeros((H, W), dtype=float)
    img2 = np.zeros((H, W), dtype=float)
    
    img1_1 = add_intensity_subpixel(img1, d, projected_points_cam1[:,0], projected_points_cam1[:,1], delta_x, delta_y)
    img2_1 = add_intensity_subpixel(img2, d, projected_points_cam2[:,0], projected_points_cam2[:,1], delta_x, delta_y)
    
    img1_1 = (img1_1 / np.max(img1_1) * 30).astype(np.uint8)
    img2_1 = (img2_1 / np.max(img2_1) * 30).astype(np.uint8)
    
    return img1_1, img2_1, trajectory_3d

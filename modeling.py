import cv2
import random

import numpy as np
import matplotlib.pyplot as plt

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
    '''
    vx0 = -v0 * np.cos(alpha)
    vy0 = v0 * np.sin(alpha)
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

def calculate_rotated_rotation_matrix(theta):
    # 旋转角度（弧度）
    angle_rad = np.radians(theta)
    print(theta/np.pi*180 )

    original_rotation_matrix = np.array([[1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.]])
    u_x, u_y, u_z = 1., 0., 0.
    # 计算旋转矩阵
    rotation_matrix = np.array([[np.cos(angle_rad) + u_x**2 * (1 - np.cos(angle_rad)), u_x * u_y * (1 - np.cos(angle_rad)) - u_z * np.sin(angle_rad), u_x * u_z * (1 - np.cos(angle_rad)) + u_y * np.sin(angle_rad)],
                                [u_y * u_x * (1 - np.cos(angle_rad)) + u_z * np.sin(angle_rad), np.cos(angle_rad) + u_y**2 * (1 - np.cos(angle_rad)), u_y * u_z * (1 - np.cos(angle_rad)) - u_x * np.sin(angle_rad)],
                                [u_z * u_x * (1 - np.cos(angle_rad)) - u_y * np.sin(angle_rad), u_z * u_y * (1 - np.cos(angle_rad)) + u_x * np.sin(angle_rad), np.cos(angle_rad) + u_z**2 * (1 - np.cos(angle_rad))]])

    # 计算旋转后的相机旋转矩阵
    rotated_rotation_matrix = np.dot(rotation_matrix, original_rotation_matrix)

    return rotated_rotation_matrix


def get_simulated_image(parameters: ModelingParabolaParameters, H, W):
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
    'R': np.array([[1., 0., 0.],
                   [0, 1., 0.],
                   [0., 0., 1.]]),
    'T': np.array([0., 0., 0.]),
    'K': np.array([[12900, 0, 960], [0, 12900, 600], [0, 0, 1]], dtype=float),
    'dist': np.array([0, 0, 0, 0], dtype=float)
    }
    F = random.uniform(-10, -110)
    theta = random.uniform(0, 0.5)
    # cam2 = {
    # 'R': np.array([[0.9956599184270521, -0.0019591982361651166, 0.09304562526044553],
    #                [0.0020270800581164237, 0.9999977438269337, -0.0006350491951564792],
    #                [-0.09304417114614887, 0.0008209039613070899, 0.9956616434976354]]),
    # # 'T': np.array([-100.24696716028096, 0, 0]),
    # # 'T': np.array([F, 0, 0]),
    # # 'R': np.array([[1., 0., 0.],
    # #                [0, 1., 0.],
    # #                [0., 0., 1.]]),
    
    # 'T': np.array([-43.24696716028096, 0, 0]),
    # 'K': np.array([[12900, 0, 960], [0, 12900, 600], [0, 0, 1]], dtype=float),
    # 'dist': np.array([0, 0, 0, 0], dtype=float)
    # }
    cam2 = {
    # 'R': np.array([[ 1,          0,          0,        ],
    #                 [ 0,          0.99999848, -0.00174533],
    #                 [ 0,          0.00174533,  0.99999848 ]]),
    'R': calculate_rotated_rotation_matrix(theta),
    'T': np.array([-43.24696716028096, -0.12027285856129845, -3.6467541290391603]),
    # 'T': np.array([-100.24696716028096, 0, 0]),
    # 'T': np.array([F, 0, 0]),
    # 'R': np.array([[1., 0., 0.],
    #                [0, 1., 0.],
    #                [0., 0., 1.]]),
    
    # 'T': np.array([-43.24696716028096, 0, 0]),
    'K': np.array([[12900, 0, 960], [0, 12900, 600], [0, 0, 1]], dtype=float),
    'dist': np.array([0, 0, 0, 0], dtype=float)
    }
# project the 3D points onto the image planes of cam1 and cam2
    projected_points_cam1, projected_points_cam2 = project_trajectory_3d(cam1, cam2, trajectory_3d)

    not_valid_points = len(projected_points_cam1[projected_points_cam1[:,0] < 0]) + \
                       len(projected_points_cam1[projected_points_cam1[:,0] > W]) + \
                       len(projected_points_cam1[projected_points_cam1[:,1] < 0]) + \
                       len(projected_points_cam1[projected_points_cam1[:,1] > H]) + \
                       len(projected_points_cam2[projected_points_cam1[:,0] < 0]) + \
                       len(projected_points_cam2[projected_points_cam1[:,0] > W]) + \
                       len(projected_points_cam2[projected_points_cam1[:,1] < 0]) + \
                       len(projected_points_cam2[projected_points_cam1[:,1] > H])
    
    if not_valid_points > 0:
        return None, None
   
    img1 = np.zeros((H, W), dtype=float)
    img2 = np.zeros((H, W), dtype=float)
    
    img1_1 = add_intensity_subpixel(img1, d, projected_points_cam1[:,0], projected_points_cam1[:,1], delta_x, delta_y)
    img2_1 = add_intensity_subpixel(img2, d, projected_points_cam2[:,0], projected_points_cam2[:,1], delta_x, delta_y)
    
    img1_1 = (img1_1 / np.max(img1_1) * 30).astype(np.uint8)
    img2_1 = (img2_1 / np.max(img2_1) * 30).astype(np.uint8)
    
    return img1_1, img2_1




# if __name__ == "__main__":
 
#     # Размер изображения частицы в [мм]
# #     d = 0.07 # mm
    
# #     # Координаты начального положения частицы [мм]
# #     x0 = -15
# #     y0 = 10
# #     # Начальная скорость [мм/с]
# #     v0 = -0.6 * 10**3
# #     # Угол взлета
# #     alpha = 75 / 180 * np.pi
# #     # Параметры плоскости в которой происходит левитация
# #     A = 0
# #     B = 0
# #     C = 1
# #     D = 500 
# #     # Шаг по времени для расчета координат траектории частицы [с]
# #     dt = 0.00001
# #     trajectory = calculate_trajectory(x0, y0, v0, alpha, A, B, C, D, dt = dt, max_time=0.1)

# #     trajectory_2d_x = trajectory[:, 0]
# #     trajectory_2d_y = trajectory[:, 1]
# #     trajectory_3d_z = trajectory[:, 2]

# #     # Plot trajectory
# #     # fig = plt.figure()
# #     # ax = fig.add_subplot(111, projection='3d')
# #     # ax.plot(trajectory_2d_x, trajectory_2d_y, trajectory_3d_z)
# #     # ax.set_xlabel('X')
# #     # ax.set_ylabel('Y')
# #     # ax.set_zlabel('Z')
# #     # ax.view_init(-60, -90)
# #     # plt.show()

# # # assume we have the 3D trajectory stored in variable "trajectory_3d"
# #     trajectory_3d = np.array(list(zip(trajectory_2d_x, trajectory_2d_y, trajectory_3d_z)))
# #     cam1 = {
# #     'R': np.array([[1., 0., 0.],
# #                    [0, 1., 0.],
# #                    [0., 0., 1.]]),
# #     'T': np.array([0., 0., 0.]),
# #     'K': np.array([[12900, 0, 960], [0, 12900, 600], [0, 0, 1]], dtype=float),
# #     'dist': np.array([0, 0, 0, 0], dtype=float)
# #     }
# #     cam2 = {
# #     'R': np.array([[0.9956599184270521, -0.0019591982361651166, 0.09304562526044553],
# #                    [0.0020270800581164237, 0.9999977438269337, -0.0006350491951564792],
# #                    [-0.09304417114614887, 0.0008209039613070899, 0.9956616434976354]]),
# #     'T': np.array([-43.24696716028096, -0.12027285856129845, -3.6467541290391603]),
# #     'K': np.array([[12900, 0, 960], [0, 12900, 600], [0, 0, 1]], dtype=float),
# #     'dist': np.array([0, 0, 0, 0], dtype=float)
# #     }
# # # project the 3D points onto the image planes of cam1 and cam2
# #     projected_points_cam1, projected_points_cam2 = project_trajectory_3d(cam1, cam2, trajectory_3d)

# #     IMAGE_WIDTH = 1920
# #     IMAGE_HEIGHT = 1280

# #     # TODO: Смоделировать дискретизацию с учетом размера частицы в пикселях
# #     modeled_image1 = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=float)
# #     modeled_image2 = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=float)

# #     x0_list = []
# #     y0_list = []
# #     x2_list = []
# #     y2_list = []

# #     # Размер пикселя матрицы
#     delta_x = 5*10**-3 # mm
#     delta_y = 5*10**-3 # mm

# #     brightness1 = add_intensity_subpixel(modeled_image1, d, projected_points_cam1[:,0], projected_points_cam1[:,1], delta_x, delta_y)

# #     brightness2 = add_intensity_subpixel(modeled_image2, d, projected_points_cam2[:,0], projected_points_cam2[:,1], delta_x, delta_y)

#     # fig, axes = plt.subplots(nrows=1, ncols=2)
#     # axes[0].imshow(brightness1)    
#     # axes[0].set_title('Brightness 1')
#     # axes[1].imshow(brightness2)
#     # axes[1].set_title('Brightness 2')
#     # plt.show()
#     brightness = get_simulated_image(d = 0.07, x0 = -15, y0 = 10, v0 = -0.6 * 10**3, alpha = 75 / 180 * np.pi, A = 0, B = 0, 
#                         C = 1, D = 500, dt = 0.00001, H = 1280, W = 1920)
#     brightness = brightness / np.max(brightness) * 30
#     # brightness1 = brightness1 / np.max(brightness1) * 30
#     # brightness2 = brightness2 / np.max(brightness2) * 30
   
#     cv2.imwrite('1.png', brightness)
    # cv2.imwrite('2.png', brightness2)

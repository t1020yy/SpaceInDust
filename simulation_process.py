import pickle
import datetime
import random
from typing import Callable, List
from matplotlib import pyplot as plt

import cv2
import numpy as np

from particle_track import Particle_track
from modeling import get_simulated_image, get_a_b_c
from modeling_parameters import ModelingParabolaParameters
from image_processing import display_processing_results, find_corresponding_component, get_connected_components, preprocess_image


def find_matched_component(img1, img2):
    preproc_img1, binary_img1 = preprocess_image(img1, 5, do_morph=False, img_to_sub=np.zeros((img1.shape), dtype=np.uint8))
    preproc_img2, binary_img2 = preprocess_image(img2, 5, do_morph=False, img_to_sub=np.zeros((img2.shape), dtype=np.uint8))

    con_components1 = get_connected_components(preproc_img1, binary_img1)
    con_components2 = get_connected_components(preproc_img2, binary_img2)

    cur_component, matched_component = find_corresponding_component(con_components1, con_components2, 0, preproc_img1, binary_img1, img1.shape[1])

    return con_components1, con_components2, cur_component, matched_component
    

def process_images(simulation_parameters, img1, img2, display_processing=True):

    con_components1, con_components2, cur_component, matched_component = find_matched_component(img1, img2)

    if display_processing:
        key = display_processing_results(
                    img1,
                    img2,
                    con_components1,
                    con_components2,
                    cur_component, 
                    matched_component,
                    'Modeled image for cam #1',
                    'Modeled image for cam #2',
                    True,
                    wait_time=500
                )

    cameraMatrix1 = simulation_parameters.cam1_K
    distCoeffs1 = simulation_parameters.cam1_dist
    cameraMatrix2 = simulation_parameters.cam2_K
    distCoeffs2 = simulation_parameters.cam2_dist
    R = simulation_parameters.cam2_R
    T = simulation_parameters.cam2_T   

    P1 = cameraMatrix1 @ np.hstack((np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 0, 0]]).T))
    P2 = cameraMatrix2 @ np.hstack((R, np.array([T]).T))

    if matched_component is not None:
        particle = Particle_track(matched_component, cur_component, P1, P2)
        particle.particle_radius = 100e-06
        particle.particle_density = 2200
        particle.voltage = 2500
        particle.grid_height = simulation_parameters.y_start_trajectory
        particle.image_name = "Modeled image"

        return particle
    else:
        return None
    

def process_2d_points_straight(simulation_parameters, trajectory_2d_cam1, trajectory_2d_cam2):
    
    cameraMatrix1 = simulation_parameters.cam1_K
    distCoeffs1 = simulation_parameters.cam1_dist
    cameraMatrix2 = simulation_parameters.cam2_K
    distCoeffs2 = simulation_parameters.cam2_dist
    R = simulation_parameters.cam2_R
    T = simulation_parameters.cam2_T   

    P1 = cameraMatrix1 @ np.hstack((np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 0, 0]]).T))
    P2 = cameraMatrix2 @ np.hstack((R, np.array([T]).T))

    points1 = np.array(trajectory_2d_cam1, dtype=float)
    points2 = np.array(trajectory_2d_cam2, dtype=float)
    points_4d = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points_3d = cv2.convertPointsFromHomogeneous(points_4d.T)

    return points_3d[:, 0, :]


def plot_parabola_histogram(parabola_image_parameters, target_count_per_bin = 2):

    heights = [params[0] for params in parabola_image_parameters]
    bin_edges = np.linspace(min(heights), max(heights), len(parabola_image_parameters)//target_count_per_bin + 1)
    counts, _ = np.histogram(heights, bins=bin_edges)
    plt.bar(bin_edges[:-1], counts, width=(bin_edges[1] - bin_edges[0]), edgecolor='black', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Values in Bins')
    plt.show()

# def adjust_parameters(parabola_image_parameters, generated_parameters, generate_simulation_parameters, check_parabola_parameters=None):
#     # 统计每个区间内的抛物线数量
#     target_count_per_bin = 2
#     heights = [params[0] for params in parabola_image_parameters]
#     bin_edges = np.linspace(min(heights), max(heights), len(parabola_image_parameters)//target_count_per_bin + 1)
#     current_counts, _ = np.histogram(heights, bins=bin_edges)
#     new_simulation_parameters = []

#     # 调整区间内的样本数量
#     for i in range(len(bin_edges) - 1):
#         current_count = current_counts[i]
#         target_count = target_count_per_bin
#         if current_count > target_count:
#             params_in_bin = [simultaion_parameter for parabola_parameter, simultaion_parameter in zip(parabola_image_parameters, generated_parameters)
#                              if parabola_parameter[0] >= bin_edges[i] and parabola_parameter[0] < bin_edges[i+1]]
#             # 随机删除多余的样本
#             if params_in_bin:
#                 random.shuffle(params_in_bin)
#             new_simulation_parameters.extend(params_in_bin[:target_count])

#         elif current_count < target_count:
#             #先把原有的加入
#             params_in_bin = [simultaion_parameter for parabola_parameter, simultaion_parameter in zip(parabola_image_parameters, generated_parameters)
#                              if parabola_parameter[0] >= bin_edges[i] and parabola_parameter[0] < bin_edges[i+1]]
#             new_simulation_parameters.extend(params_in_bin)
#             # 生成新的样本并添加到模拟参数列表中
#             num_samples_to_add = target_count - current_count
#             max_attempts = 100
#             attempts = 0
#             for _ in range(num_samples_to_add):
#                 simulation_parameters = []
#                 while  attempts < max_attempts:
#                     new_simulation_parameters_candidate  = generate_simulation_parameters(simulation_parameters)
#                     for simulation_parameter in new_simulation_parameters_candidate :
#                         result = get_simulated_image(simulation_parameter)
#                         if len(result) == 6:
#                             img1, img2, _, parabola_height, parabola_width, branches_height_ratio = result
#                             if check_parabola_parameters is None or check_parabola_parameters(parabola_height, parabola_width, branches_height_ratio):
#                                 if bin_edges[i] <= parabola_height < bin_edges[i + 1]:
#                                     new_simulation_parameters.append(simulation_parameter)
#                                     break    
#                             attempts += 1
#                     else:
#                         continue
#                     if attempts == max_attempts:
#                         print("Max attempts reached. Could not find suitable simulation parameters.")
#                     break        

#         elif current_count == target_count:
#             params_in_bin = [simultaion_parameter for parabola_parameter, simultaion_parameter in zip(parabola_image_parameters, generated_parameters)
#                              if parabola_parameter[0] >= bin_edges[i] and parabola_parameter[0] < bin_edges[i+1]]
#             new_simulation_parameters.extend(params_in_bin)

#     return new_simulation_parameters    

def simultaion(generate_simulation_parameters: Callable[[List[ModelingParabolaParameters]], List[ModelingParabolaParameters]], check_parabola_parameters=None):
    
    simulation_parameters = []
    simulation_parameters_to_remove = []
    simulation_parameters = generate_simulation_parameters(simulation_parameters)

    simulation_parameters_count = len(simulation_parameters)
    processing_results = []

    parabola_image_parameters = []
    simulation_parabola_parameters = []
    generated_parameters=[]
    plane_parameters = []

    try:
        while len(processing_results) < simulation_parameters_count:
            for simulation_parameter in simulation_parameters:
                result = get_simulated_image(simulation_parameter)

                img1, img2, trajectory_2d_cam1, trajectory_2d_cam2, trajectory_3d, parabola_height, parabola_width, branches_height_ratio = result                                            
                
                if img1 is not None and img2 is not None:
                    # Проверяем изображение параболы на параметры
                    if check_parabola_parameters is not None and not check_parabola_parameters(parabola_height, parabola_width, branches_height_ratio):
                        simulation_parameters_to_remove.append(simulation_parameter)
                        continue
                    
                    points_3d = process_2d_points_straight(simulation_parameter, trajectory_2d_cam1, trajectory_2d_cam2)

                    particle = process_images(simulation_parameter, img1, img2)

                    if particle is not None:
                        # Если был получен результат обработки, то сохраняем все в список результатов
                        a, b, c = get_a_b_c(
                            simulation_parameter.start_speed,
                            simulation_parameter.start_angle,
                            simulation_parameter.x_start_trajectory,
                            simulation_parameter.y_start_trajectory,
                            g = 9.81*10**3
                        )

                        G = np.ones((trajectory_3d.shape[0], 3))
                        G[:,0] = trajectory_3d[:,0]  #X
                        G[:,1] = trajectory_3d[:,1]  #Y
                        Z = trajectory_3d[:,2]
                        (plane_a, plane_b, plane_c), resid, rank, s = np.linalg.lstsq(G, Z, rcond=None) 
                        plane_parameters.append((plane_a, plane_b, plane_c))
                      

                        simulation_parabola_parameters.append((a, b, c))

                        parabola_image_parameters.append((parabola_height, parabola_width, branches_height_ratio))
                        generated_parameters.append(simulation_parameter)

                        processing_results.append((particle.Alpha, particle.V0,  particle.surface, particle.parabola, particle.parameters))
                        # print("Particle Surface Parameters:", particle.surface)
                        print(f'Сохранено {len(processing_results)} треков')

                        if len(processing_results) >= simulation_parameters_count:
                            print(f'Получено заданное количество результатов моделирования. Выход из цикла...')
                            break
                    else:
                        simulation_parameters_to_remove.append(simulation_parameter)
                        print(f'Не удалось найти стереопару параболы')
                else:
                    simulation_parameters_to_remove.append(simulation_parameter)
                    print(f'Точки параболы вышли за границу изображения')      
            # Удаление параметров, которые не подошли для моделирования
            for simulation_parameter in simulation_parameters_to_remove:
                simulation_parameters.remove(simulation_parameter)                
            simulation_parameters_to_remove.clear()

            # Создание новых параметров для замены удаленных
            if len(processing_results) < simulation_parameters_count:
                simulation_parameters = generate_simulation_parameters(simulation_parameters)
                if len(simulation_parameters) == 0:
                    break

            # if len(processing_results) == simulation_parameters_count:
            #     new_simulation_parameters = adjust_parameters(parabola_image_parameters, generated_parameters, generate_simulation_parameters, check_parabola_parameters = check_parabola_parameters)
            #     if not new_simulation_parameters:
            #         print("Warning: adjust_parameters returned an empty list.")
            #     else:
            #         processing_results = []
            #         simulation_parameters = new_simulation_parameters
            #         n+=1
                              
    except Exception as ex:
        print(f'Произошла ошибка: {ex}')

    finally:
        file_name = f'modeling_results_{datetime.datetime.now(): %Y-%m-%d_%H-%M-%S}.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump((processing_results, parabola_image_parameters, simulation_parabola_parameters, generated_parameters,plane_parameters), f)
        print(f'Результаты сохранены в файл {file_name}')
            
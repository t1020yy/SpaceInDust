import pickle
import datetime
from typing import Callable, List

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

    cur_component, matched_component = find_corresponding_component(con_components1, con_components2, 0, preproc_img1, binary_img1, 600)

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
    

def simultaion(generate_simulation_parameters: Callable[[List[ModelingParabolaParameters]], List[ModelingParabolaParameters]], check_parabola_parameters=None):
    
    simulation_parameters = []
    simulation_parameters_to_remove = []
    simulation_parameters = generate_simulation_parameters(simulation_parameters)
    simulation_parameters_count = len(simulation_parameters)
    processing_results = []

    parabola_image_parameters = []
    simulation_parabola_parameters = []
    try:
        while len(processing_results) < simulation_parameters_count:
            for simulation_parameter in simulation_parameters:
                result = get_simulated_image(simulation_parameter)
                if len(result) == 6: 
                    img1, img2, _, parabola_height, parabola_width, branches_height_ratio = result
                else:
                    simulation_parameters_to_remove.append(simulation_parameter)
                    continue
                if len(processing_results) >= simulation_parameters_count:
                    break
                # Проверяем изображение параболы на параметры
                if check_parabola_parameters is not None and not check_parabola_parameters(parabola_height, parabola_width, branches_height_ratio):
                    simulation_parameters_to_remove.append(simulation_parameter)
                    continue

                if img1 is not None and img2 is not None:
                    
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

                        simulation_parabola_parameters.append((a, b, c))

                        parabola_image_parameters.append((parabola_height, parabola_width, branches_height_ratio))

                        processing_results.append((particle.Alpha, particle.V0,  particle.surface, particle.parabola, particle.parameters))
                        print(f'Сохранено {len(processing_results)} треков')
                    else:
                        simulation_parameters_to_remove.append(simulation_parameter)
                        print(f'Не удалось найти стереопару параболы')
                else:
                    simulation_parameters_to_remove.append(simulation_parameter)
                    print(f'Не удалось смоделировать')
            # Удаление параметров, которые не подошли для моделирования
            for simulation_parameter in simulation_parameters_to_remove:
                simulation_parameters.remove(simulation_parameter)                
            simulation_parameters_to_remove.clear()

            # Создание новых параметров для замены удаленных
            if len(processing_results) < simulation_parameters_count:
                simulation_parameters = generate_simulation_parameters(simulation_parameters)
                if len(simulation_parameters) == 0:
                    break
            
        # Доделать - если есть новые параметры, то нужно продолжить моделирование в цикле

    except Exception as ex:
        print(f'Произошла ошибка: {ex}')

    finally:
        file_name = f'modeling_results_{datetime.datetime.now(): %Y-%m-%d_%H-%M-%S}.pickle'
        with open(file_name, 'wb') as f:
            pickle.dump((simulation_parameters, processing_results, parabola_image_parameters, simulation_parabola_parameters), f)
        print(f'Результаты сохранены в файл {file_name}')
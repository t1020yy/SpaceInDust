import pickle
import datetime
import random

import numpy as np
from matplotlib import pyplot as plt

from modeling import get_simulated_image
from particle_track import Particle_track
from modeling_parameters import ModelingParabolaParameters
from image_processing import display_processing_results, find_corresponding_component, get_connected_components, preprocess_image


def process_images(img1, img2):
    preproc_img1, binary_img1 = preprocess_image(img1, 5, do_morph=False, img_to_sub=np.zeros((img1.shape), dtype=np.uint8))
    preproc_img2, binary_img2 = preprocess_image(img2, 5, do_morph=False, img_to_sub=np.zeros((img2.shape), dtype=np.uint8))

    con_components1 = get_connected_components(preproc_img1, binary_img1)
    con_components2 = get_connected_components(preproc_img2, binary_img2)

    cur_component, matched_component = find_corresponding_component(con_components1, con_components2, 0, preproc_img1, binary_img1, 600)

    return con_components1, con_components2, cur_component, matched_component
    

if __name__ == "__main__":
    modeling_number = np.arange(10)
    expose_time_starts = 0.050 - np.random.rand(modeling_number.shape[0]) * 0.100
    parameters = []
    result = []
    result_params = []
    cams_trans_vec_x = [] 

    cams_trans_vec_x_values = np.linspace(5,15,30)
    cams_rot_y_values = np.linspace(0,0.7,30)
    for vec_x in cams_trans_vec_x_values:
        for rot_ys in cams_rot_y_values:
            for i in modeling_number:
                param = ModelingParabolaParameters()
                param.x_start_trajectory = -12 #mm
                param.y_start_trajectory = 15 #mm
                param.start_speed = 0.6 * 10**3 #mm/s
                param.expose_time_start = expose_time_starts[i] 
                param.expose_time = 0.100
                # param.particle_diameter = 0.01 + 0.01 * (i // 10)
                param.particle_diameter = 0.05
                # param.start_angle = 40 + 0.8 * (i // 10)
                # param.start_speed = (0.35 + 0.01 * (i // 10)) * 10**3 #mm/s
                param.cams_trans_vec_x = vec_x

                param.cams_rot_y = rot_ys

                parameters.append(param)


    for param in parameters:

        img1, img2, _ = get_simulated_image(param)
        con_components1, con_components2, cur_component, matched_component = process_images(img1, img2)
        if img1 is not None and img2 is not None:
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

            cameraMatrix1 = param.cam1_K
            distCoeffs1 = param.cam1_dist
            cameraMatrix2 = param.cam2_K
            distCoeffs2 = param.cam2_dist
            R = param.cam2_R
            T = param.cam2_T   

            P1 = cameraMatrix1 @ np.hstack((np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 0, 0]]).T))
            P2 = cameraMatrix2 @ np.hstack((R, np.array([T]).T))

            if matched_component is not None:
                particle = Particle_track(matched_component, cur_component, P1, P2)
                particle.particle_radius = 100e-06
                particle.particle_density = 2200
                particle.voltage = 2500
                particle.grid_height = param.y_start_trajectory
                particle.image_name = "Modeled image"

                result.append(particle)
                result_params.append((particle.Alpha, particle.V0, particle.surface, particle.parabola, particle.parameters))
                print(f'Сохранено {len(result)} треков')
            else:
                result.append(None)
                print(f'Не удалось найти стереопару параболы')
        else:
            result.append(None)
            print(f'Не удалось смоделировать')

                                        
    file_name = f'modeling_results_{datetime.datetime.now(): %Y-%m-%d_%H-%M-%S}.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump((parameters, result_params), f)

    print(f'Результаты сохранены в файл {file_name}')

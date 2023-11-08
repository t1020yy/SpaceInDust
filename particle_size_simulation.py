import pickle
import datetime
import random

import numpy as np

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
    modeling_number = np.arange(1000)
    parameters = []
    result_params = []
    cams_trans_vec_x = [] 
    kk_values = []
    h_values = []

    # cams_trans_vec_x_values = np.linspace(10, 30, 30)
    # cams_rot_y_values = np.linspace(-10, 0, 30)

    # for vec_x in cams_trans_vec_x_values:
    #     for rot_ys in cams_rot_y_values:
    #         for i in modeling_number:
    #             param = ModelingParabolaParameters()
    #             param.x_start_trajectory = -12 #mm
    #             param.y_start_trajectory = 15 #mm
    #             # param.start_speed = 0.6 * 10**3 #mm/s
    #             param.expose_time_start = 0.050 - random.random() * 0.100
    #             param.expose_time = 0.100
    #             # param.particle_diameter = 0.01 + 0.01 * (i // 10)
    #             param.particle_diameter = 0.05
    #             param.start_angle = 55 + 30 * random.random()
    #             param.start_speed = (0.35 + 0.25 * random.random()) * 10**3 #mm/s
    #             param.cams_trans_vec_x = vec_x

    #             param.cams_rot_y = rot_ys

    #             parameters.append(param)
     
    for i in modeling_number:
        param = ModelingParabolaParameters()
        param.x_start_trajectory = -12 #mm
        param.y_start_trajectory = 15 #mm
        # param.start_speed = 0.6 * 10**3 #mm/s
        param.expose_time_start = 0.050 - random.random() * 0.100
        param.expose_time = 0.100
                # param.particle_diameter = 0.01 + 0.01 * (i // 10)
        param.particle_diameter = 0.05
        # param.start_angle = 55 + 30 * random.random()
        param.start_speed = (0.35 + 0.25 * random.random()) * 10**3 #mm/s
        # param.cams_trans_vec_x = vec_x

        # param.cams_rot_y = rot_ys
        parameters.append(param)
    

    for param in parameters:

        img1, img2, _, kk, h = get_simulated_image(param)
        if kk is not None and h is not None:
            kk_values.append(kk)
            h_values.append(h)
        else:
            pass

        if img1 is not None and img2 is not None:
            con_components1, con_components2, cur_component, matched_component = process_images(img1, img2)
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

                result_params.append((particle.Alpha, particle.V0, particle.surface, particle.parabola, particle.parameters))
                print(f'Сохранено {len(result_params)} треков')
            else:
                result_params.append(None)
                print(f'Не удалось найти стереопару параболы')
        else:
            result_params.append(None)
            print(f'Не удалось смоделировать')
        
                                        
    file_name = f'modeling_results_{datetime.datetime.now(): %Y-%m-%d_%H-%M-%S}.pickle'
    with open(file_name, 'wb') as f:
        pickle.dump((parameters, result_params, kk_values, h_values), f)

    print(f'Результаты сохранены в файл {file_name}')

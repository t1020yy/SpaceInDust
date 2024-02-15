import math
import random
from typing import List

import numpy as np
from modeling import get_simulated_image

from simulation_process import simultaion
from modeling_parameters import ModelingParabolaParameters


def check_parabola_parameters(parabola_height, parabola_width, branches_height_ratio):
    if parabola_height < 250 or parabola_width < 200 or (branches_height_ratio < 0.2 or branches_height_ratio > 0.85):
        return False
    else:
        return True


def generate_simulation_parameters(simulation_parameters: List[ModelingParabolaParameters]):

    parameters_to_generate_number = PARAMS_TO_GENERATE - len(simulation_parameters)

    generated_parameters = []

    if parameters_to_generate_number < PARAMS_DIFF_THRESHOLD:
        return generated_parameters

    for _ in range(parameters_to_generate_number):
        generating_parameter = ModelingParabolaParameters()
        generating_parameter.x_start_trajectory = -12 #mm
        generating_parameter.y_start_trajectory = 15 #mm
        # param.start_speed = 0.6 * 10**3 #mm/s
        generating_parameter.expose_time_start = 0.050 - random.random() * 0.100
        generating_parameter.expose_time = 0.100
        # param.particle_diameter = 0.01 + 0.01 * (i // 10)
        generating_parameter.particle_diameter = 0.05
        generating_parameter.start_angle = 55 + 30 * random.random()
        generating_parameter.start_speed = (0.35 + 0.25 * random.random()) * 10**3 #mm/s
        # print("param.start_speed", param.start_speed)
        # param.cams_trans_vec_x = vec_x
        generated_parameters.append(generating_parameter)
    
    return generated_parameters


def adjust_parameters(parabola_image_parameters, generated_parameters, generate_simulation_parameters, target_count_per_bin, check_parabola_parameters = check_parabola_parameters):
    # 统计每个区间内的抛物线数量
    heights = [params[0] for params in parabola_image_parameters]
    bin_edges = np.linspace(min(heights), max(heights), len(parabola_image_parameters)//target_count_per_bin + 1)
    current_counts, _ = np.histogram(heights, bins=bin_edges)
    new_simulation_parameters = []
    # 调整区间内的样本数量
    for i in range(len(bin_edges) - 1):
        current_count = current_counts[i]
        target_count = target_count_per_bin
        added_samples = 0
        attempts = 0 

        if current_count != target_count:
            # 随机删除多余的样本
            if current_count > target_count:
                #随机删除n个高宽对应的参数值，将删除后的参数放到new_simulation_parameters中
                params_in_bin = []
                
                for parabola_parameter, simultaion_parameter in zip(parabola_image_parameters, generated_parameters):
                    if parabola_parameter[0] >= bin_edges[i] and parabola_parameter[0] < bin_edges[i+1]:
                        params_in_bin.append(simultaion_parameter)
                
                new_simulation_parameters.append(random.shuffle(params_in_bin)[:target_count + 1])
            elif current_count < target_count:
                # 生成新的样本并添加到模拟参数列表中
                num_samples_to_add = target_count - current_count
                max_attempts = 100
                for _ in range(num_samples_to_add):
                    added_samples = 0
                    while added_samples < num_samples_to_add and attempts < max_attempts:
                        simulation_parameters = []
                        simulation_parameters = generate_simulation_parameters(simulation_parameters)
                        for simulation_parameter in simulation_parameters:
                            result = get_simulated_image(simulation_parameter)
                            if len(result) == 6: 
                                img1, img2, _, parabola_height, parabola_width, branches_height_ratio = result
                            if check_parabola_parameters is None or check_parabola_parameters(parabola_height, parabola_width, branches_height_ratio):
                                if bin_edges[i] <= parabola_height < bin_edges[i + 1]:
                                    new_simulation_parameters.append(simulation_parameter)
                                    added_samples += 1
                        attempts +=1
                    if attempts == max_attempts:
                        print("Max attempts reached. Could not find suitable simulation parameters.")
    return new_simulation_parameters

if __name__ == "__main__":

    PARAMS_TO_GENERATE = 100
    PARAMS_DIFF_THRESHOLD_PERCENTAGE = 1 # %
    PARAMS_DIFF_THRESHOLD = math.floor(PARAMS_TO_GENERATE * PARAMS_DIFF_THRESHOLD_PERCENTAGE / 100)

    simultaion(generate_simulation_parameters, check_parabola_parameters = check_parabola_parameters)
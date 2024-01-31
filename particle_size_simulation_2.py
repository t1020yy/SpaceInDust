import random
from typing import List

from simulation_process import simultaion
from modeling_parameters import ModelingParabolaParameters


def generate_simulation_parameters(simulation_parameters: List[ModelingParabolaParameters]):

    parameters_to_generate_number = PARAMS_TO_GENERATE - len(simulation_parameters)    

    generated_parameters = []

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

    if len(simulation_parameters) > 0:
        simulation_parameters.extend(generated_parameters)
        return simulation_parameters
    
    return generated_parameters


if __name__ == "__main__":

    PARAMS_TO_GENERATE = 300

    simultaion(generate_simulation_parameters)
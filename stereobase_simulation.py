import numpy as np
from matplotlib import pyplot as plt

from image_processing import main_processing_loop
from modeling_parameters import ModelingParabolaParameters

stereo_base = np.linspace(-43, -44, num=5)

parameters = []

for base in stereo_base:
    param = ModelingParabolaParameters()
    param.cams_trans_vec_x = base

    parameters.append(param)

result = main_processing_loop(parameters)

angle_err = []
velocity_err = []
base = []

for i in range(len(parameters)):
    if result[i] is not None:
        parameter = parameters[i]
        particle = result[i]

        angle_err.append(parameter.start_angle / np.pi * 180 - particle.Alpha)
        velocity_err.append(parameter.start_speed - particle.V0)
        base.append(parameter.cams_trans_vec_x)    
    
plt.plot(base, angle_err, 'bo')
plt.show()

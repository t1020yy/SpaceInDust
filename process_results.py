import pickle

import numpy as np
from matplotlib import pyplot as plt


FILE_NAME = 'modeling_results_ 2023-10-31_15-06-38.pickle'

with open(FILE_NAME, 'rb') as f:
    parameters, result = pickle.load(f)

angle_err = []
velocity_err = []
diameter = []

for i in range(len(parameters)):
    if result[i] is not None:
        parameter = parameters[i]
        particle = result[i]

        angle_err.append(parameter.start_angle - np.abs(particle[0]))
        velocity_err.append(parameter.start_speed * 10**-3 - particle[1])
        diameter.append(parameter.particle_diameter)

plt.subplot(121)
plt.plot(diameter, angle_err, 'bo')
plt.grid()
plt.subplot(122)
plt.plot(diameter, velocity_err, 'bo')
plt.grid()
plt.show()
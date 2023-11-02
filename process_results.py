import pickle

import numpy as np
from matplotlib import pyplot as plt


FILE_NAME = 'modeling_results_ 2023-11-02_16-22-14.pickle'

with open(FILE_NAME, 'rb') as f:
    parameters, result = pickle.load(f)

angle_err = []
velocity_err = []
diameter = []
start_angle = []
start_speed = []
cams_trans_vec_x_values = []
angle_between_cameras_values = []

for i in range(len(parameters)):
    # if result[i] is not None:
    if i < len(result) and result[i] is not None:
        parameter = parameters[i]
        particle = result[i]

        angle_err.append(parameter.start_angle - np.abs(particle[0]))
        velocity_err.append(parameter.start_speed * 10**-3 - particle[1])
        # diameter.append(parameter.particle_diameter)
        start_angle.append(parameter.start_angle)
        # start_speed.append(parameter.start_speed)
        cams_trans_vec_x_values.append(parameter.cams_trans_vec_x)

        relative_rotation = np.dot(np.linalg.inv(parameter.cam1_R), parameter.cam2_R)

        # 将相对旋转矩阵转换为欧拉角
        relative_euler_angles = np.degrees(np.array([np.arctan2(relative_rotation[2, 1], relative_rotation[2, 2]),
                                                    np.arctan2(-relative_rotation[2, 0], np.sqrt(relative_rotation[2, 1] ** 2 + relative_rotation[2, 2] ** 2)),
                                                    np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0])]))

        # 计算夹角
        angle_between_cameras = np.linalg.norm(relative_euler_angles)
        angle_between_cameras_values.append(angle_between_cameras)



plt.subplot(121)
plt.plot(start_angle, angle_err, 'bo')
plt.grid()
plt.subplot(122)
plt.plot(start_angle, velocity_err, 'bo')
plt.grid()
plt.show()

# plt.subplot(121)
# plt.plot(angle_between_cameras_values, angle_err, 'bo')
# plt.grid()
# plt.subplot(122)
# plt.plot(angle_between_cameras_values, velocity_err, 'bo')
# plt.grid()
# plt.show()

# plt.subplot(121)
# plt.scatter(cams_trans_vec_x_values, angle_between_cameras_values, c=angle_err, cmap='viridis')
# plt.xlabel('param.cams_trans_vec_x')
# plt.ylabel('angle_between_cameras')
# plt.colorbar(label='angle_err')
# plt.grid()
# plt.subplot(122)
# plt.scatter(cams_trans_vec_x_values, angle_between_cameras_values, c=velocity_err, cmap='viridis')
# plt.xlabel('param.cams_trans_vec_x')
# plt.ylabel('angle_between_cameras')

# plt.colorbar(label='velocity_err')
# plt.grid()
# plt.show()
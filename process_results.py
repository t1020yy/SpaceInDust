import pickle

import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit


FILE_NAME = 'modeling_results_ 2023-11-09_00-35-47.pickle'

with open(FILE_NAME, 'rb') as f:
    parameters, result, kk_values, h_values= pickle.load(f)

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

        angle_err.append(abs(parameter.start_angle - np.abs(particle[0])))
        velocity_err.append(abs(parameter.start_speed * 10**-3 - particle[1]))
        # diameter.append(parameter.particle_diameter)
        start_angle.append(parameter.start_angle)
        start_speed.append(parameter.start_speed)
        cams_trans_vec_x_values.append(parameter.cams_trans_vec_x)

        relative_rotation =parameter.cam2_R
        relative_rotation_y = np.degrees(np.arctan2(relative_rotation[2, 0], relative_rotation[0, 0])) 
        angle_between_cameras = np.abs(relative_rotation_y)
        angle_between_cameras_values.append(angle_between_cameras)

plt.subplot(121)
plt.plot(start_speed, angle_err, 'bo')
plt.xlabel('start_speed', fontsize = 20)
plt.ylabel('Angle error (°)', fontsize = 20)
plt.xticks(fontsize = 20)  
plt.yticks(fontsize = 20)
plt.grid()
plt.subplot(122)
plt.plot(start_speed, velocity_err, 'bo')
plt.xlabel('start_speed', fontsize = 20)
plt.ylabel('Velocity error (m/s)', fontsize = 20)
plt.xticks(fontsize = 20)  
plt.yticks(fontsize = 20)
plt.grid()
plt.show()

plt.subplot(121)
plt.plot(kk_values, angle_err, 'bo')
plt.xlabel('kk', fontsize = 20)
plt.ylabel('Angle error (°)', fontsize = 20)
plt.xticks(fontsize = 20)  
plt.yticks(fontsize = 20)
plt.grid()
plt.subplot(122)
plt.plot(kk_values, velocity_err, 'bo')
plt.xlabel('kk', fontsize = 20)
plt.ylabel('Velocity error (m/s)', fontsize = 20)
plt.xticks(fontsize = 20)  
plt.yticks(fontsize = 20)
plt.grid()
plt.show()

plt.subplot(121)
plt.plot(h_values, angle_err, 'bo')
plt.xlabel('h(mm)', fontsize = 20)
plt.ylabel('Angle error (°)', fontsize = 20)
plt.xticks(fontsize = 20)  
plt.yticks(fontsize = 20)
plt.grid()
plt.subplot(122)
plt.plot(h_values, velocity_err, 'bo')
plt.xlabel('h(mm)', fontsize = 20)
plt.ylabel('Velocity error (m/s)', fontsize = 20)
plt.xticks(fontsize = 20)  
plt.yticks(fontsize = 20)
plt.grid()
plt.show()

#计算移动距离，旋转向量与角度差或速度差的关系图
# x = np.array(cams_trans_vec_x_values)
# y = np.array(angle_between_cameras_values)
# z = np.array(angle_err)
# plt.subplot(121)
# contour = plt.tricontourf(x, y, z, cmap='viridis')
# cbar = plt.colorbar(contour)
# cbar.ax.tick_params(labelsize=16) 
# plt.xlabel('cams_trans_vec_x (mm)', fontsize = 16)
# plt.ylabel('angle_between_cameras (°)', fontsize = 16)
# plt.title('angle_err', fontsize = 16)
# plt.xticks(fontsize = 16)  
# plt.yticks(fontsize = 16)
# plt.grid()
# plt.subplot(122)
# z = np.array(velocity_err)
# contour = plt.tricontourf(x, y, z, cmap='viridis')
# cbar = plt.colorbar(contour)
# cbar.ax.tick_params(labelsize=16) 
# plt.xlabel('cams_trans_vec_x (mm)', fontsize = 16)
# plt.ylabel('angle_between_cameras(°)', fontsize = 16)
# plt.title('velocity_err', fontsize = 16)
# plt.xticks(fontsize = 16)  
# plt.yticks(fontsize = 16)
# plt.grid()
# plt.show()

# diameter = np.array(diameter)
# # start_speed = np.array(start_speed)
# start_angle = np.array(start_angle)
# angle_err = np.array(angle_err)
# velocity_err = np.array(velocity_err)
# unique_diameters = np.unique(diameter)
# unique_start_speed = np.unique(start_speed)
# unique_start_angle = np.unique(start_angle)

# mean_angle_err = [np.mean(angle_err[start_angle == d]) for d in unique_start_angle]
# mean_velocity_err = [np.mean(velocity_err[start_angle == d]) for d in unique_start_angle]
# std_dev_angle_err = [np.std(angle_err[start_angle == d]) for d in unique_start_angle]
# std_dev_velocity_err = [np.std(velocity_err[start_angle == d]) for d in unique_start_angle]



# # #计算平均值和原始数据的error bar图
# plt.subplot(121)
# plt.errorbar(unique_start_angle, mean_angle_err, yerr=std_dev_angle_err, fmt='o', color='b', ecolor='r', capsize=5)
# # plt.plot(diameter, angle_err, 'bo')
# plt.xlabel('start_angle (°)', fontsize = 20)
# plt.ylabel('Angle error (°)', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.gca().xaxis.get_label().set_ha('center')  # 设置x轴标签居中对齐
# plt.gca().yaxis.get_label().set_ha('center')
# plt.grid()
# plt.subplot(122)
# plt.errorbar(unique_start_angle, mean_velocity_err, yerr=std_dev_velocity_err, fmt='o', color='b', ecolor='r', capsize=5)
# # plt.plot(diameter, velocity_err, 'bo')
# plt.xlabel('start_angle (°)', fontsize = 20)
# plt.ylabel('Velocity error (m/s)', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.gca().xaxis.get_label().set_ha('center')  # 设置x轴标签居中对齐
# plt.gca().yaxis.get_label().set_ha('center')
# plt.grid()
# plt.show()


# #拟合后的曲线值作为平均值
# coefficients = np.polyfit(unique_diameters, mean_angle_err, 2)
# coefficients1 = np.polyfit(unique_diameters, mean_velocity_err, 2)
# polynomial = np.poly1d(coefficients)
# polynomial1 = np.poly1d(coefficients1)
# # 生成拟合曲线上的点
# x_fit = np.linspace(min(unique_diameters), max(unique_diameters), 200)
# y_fit = polynomial(x_fit)
# new_mean_angle_err = polynomial(unique_diameters)

# y_fit1 = polynomial1(x_fit)
# new_mean_velocity_err = polynomial1(unique_diameters)
# # 绘制图像时使用新的 mean_angle_err
# plt.subplot(121)
# plt.errorbar(unique_diameters, new_mean_angle_err, yerr=std_dev_angle_err, fmt='o', color='b', ecolor='r', capsize=5)
# plt.xlabel('Partice diameter (mm)', fontsize=20)
# plt.ylabel('Angle error (°)', fontsize=20)
# plt.xticks(fontsize=20)  
# plt.yticks(fontsize=20)
# plt.gca().xaxis.get_label().set_ha('center')
# plt.gca().yaxis.get_label().set_ha('center')
# plt.grid()

# plt.subplot(122)
# plt.errorbar(unique_diameters, new_mean_velocity_err, yerr=std_dev_velocity_err, fmt='o', color='b', ecolor='r', capsize=5)
# plt.xlabel('Partice diameter (mm)', fontsize=20)
# plt.ylabel('velocity_err (m/s)', fontsize=20)
# plt.xticks(fontsize=20)  
# plt.yticks(fontsize=20)
# plt.gca().xaxis.get_label().set_ha('center')
# plt.gca().yaxis.get_label().set_ha('center')
# plt.grid()
# plt.show()




# # #计算Boxplot图
# data_a = [angle_err[start_angle == d] for d in unique_start_angle]
# data_v = [velocity_err[start_angle == d] for d in unique_start_angle]
# unique_diameters = np.around(unique_diameters, 2)
# # unique_start_speed = [int(d) for d in unique_start_speed]
# unique_start_angle = [int(d) for d in unique_start_angle]
# plt.boxplot(data_a, labels=unique_start_angle)
# plt.xlabel('Partice diameter (mm)', fontsize = 20)
# plt.ylabel('Angle error (°)', fontsize = 20)

# plt.xticks(rotation=45,fontsize = 15)   
# plt.yticks(fontsize = 15)
# plt.gca().xaxis.get_label().set_ha('center')
# plt.gca().yaxis.get_label().set_ha('center')
# plt.show()

# plt.boxplot(data_v, labels=unique_start_angle)
# plt.xlabel('Partice diameter (mm)', fontsize = 20)
# plt.ylabel('Velocity error (m/s)', fontsize = 20)
# plt.xticks(rotation=45,fontsize = 15)  
# plt.yticks(fontsize = 15)
# plt.gca().xaxis.get_label().set_ha('center')  
# plt.gca().yaxis.get_label().set_ha('center')
# plt.show()

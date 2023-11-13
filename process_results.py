import pickle

import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy.optimize import curve_fit
from modeling import get_a_b_c


FILE_NAME = 'modeling_results_ 2023-11-13_20-48-04.pickle'

with open(FILE_NAME, 'rb') as f:
    parameters, result, kk_values, h_values, d_values, a_values, b_values, c_values = pickle.load(f)

angle_err = []
velocity_err = []
diameter = []
start_angle = []
start_speed = []
cams_trans_vec_x_values = []
angle_between_cameras_values = []
a_err = []
b_err = []
c_err = []
a_1_values = []
b_1_values = []
c_1_values = []
for i in range(len(parameters)):
    # if result[i] is not None:
    if i < len(result) and result[i] is not None:
        parameter = parameters[i]
        particle = result[i]
        a_1, b_1, c_1 = get_a_b_c(particle[1]*10**3, np.abs(particle[0]), parameter.x_start_trajectory, parameter.y_start_trajectory, g = 9.81*10**3)
        a_1_values.append(a_1)
        b_1_values.append(b_1)
        c_1_values.append(c_1)
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
a_err = [abs(a - a_1) for a, a_1 in zip(a_values, a_1_values)]
b_err = [abs(b - b_1) for b, b_1 in zip(b_values, b_1_values)]
c_err = [abs(c - c_1) for c, c_1 in zip(c_values, c_1_values)]

# 创建一个包含3个子图的图表
fig, axs = plt.subplots(1, 3, figsize=(16, 5))

# 绘制第一个子图
axs[0].plot(h_values, a_err, 'bo')
axs[0].set_xlabel('dd', fontsize=16)
axs[0].set_ylabel('a_err', fontsize=16)
axs[0].tick_params(axis='both', labelsize=16)
axs[0].grid()

# 绘制第二个子图
axs[1].plot(h_values, b_err, 'bo')
axs[1].set_xlabel('dd', fontsize=16)
axs[1].set_ylabel('b_err', fontsize=16)
axs[1].tick_params(axis='both', labelsize=16)
axs[1].grid()

# 绘制第三个子图
axs[2].plot(h_values, c_err, 'bo')
axs[2].set_xlabel('dd', fontsize=16)
axs[2].set_ylabel('c_err', fontsize=16)
axs[2].tick_params(axis='both', labelsize=16)
axs[2].grid()

# 调整子图的布局，避免重叠
plt.tight_layout()

# 显示图表
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))

# 绘制 a_err、b_err 和 c_err 在同一子图中
ax.plot(h_values, a_err, 'bo', label='a_err')
ax.plot(h_values, b_err, 'ro', label='b_err')
ax.plot(h_values, c_err, 'go', label='c_err')

# 设置标签和标题
ax.set_xlabel('dd', fontsize=16)
ax.set_ylabel('Errors', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid()
ax.legend()  # 添加图例

# 显示图表
plt.show()

# 创建一个包含1个子图的图表
fig, ax1 = plt.subplots(figsize=(10, 5))

# 绘制第一个数据集
line1, = ax1.plot(h_values, a_err, 'bo', label='a_err')
ax1.set_xlabel('dd', fontsize=16)
ax1.set_ylabel('a_err', color='b', fontsize=20)
ax1.tick_params(axis='both', labelsize=16)
ax1.grid()

# 创建第二个 y 轴，共享 x 轴
ax2 = ax1.twinx()

# 绘制第二个数据集
line2, = ax2.plot(h_values, b_err, 'rx', label='b_err')
ax2.set_ylabel('b_err', color='r', fontsize=20)
ax2.tick_params(axis='both', labelsize=16)

# 创建第三个 y 轴，共享 x 轴
ax3 = ax1.twinx()

# 将第三个轴移到右侧
ax3.spines['right'].set_position(('outward', 60))
line3, = ax3.plot(h_values, c_err, 'g^', label='c_err')
ax3.set_ylabel('c_err', color='g', fontsize=20)
ax3.tick_params(axis='both', labelsize=16)

# 调整子图的布局，避免重叠
plt.tight_layout()

# 创建独立的图例，并设置它们的位置
fig.legend([line1, line2, line3], ['a_err', 'b_err', 'c_err'], loc='upper right', bbox_to_anchor=(0.825, 0.95))

# 显示图表
plt.show()
# plt.subplot(121)
# plt.plot(d_values, a_err, 'bo')
# plt.xlabel('kk', fontsize = 20)
# plt.ylabel('a_err', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.grid()
# plt.subplot(122)
# plt.plot(d_values, b_err, 'bo')
# plt.xlabel('kk', fontsize = 20)
# plt.ylabel('b_err', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.grid()
# plt.show()

# plt.plot(d_values, c_err, 'bo')
# plt.xlabel('kk', fontsize = 20)
# plt.ylabel('c_err', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.grid()
# plt.show()
# kk_values = np.array(kk_values)
# a_err = np.array(a_err)
# unique_kk_values = np.unique(kk_values)
# # mean_a_err = [np.mean(angle_err[start_angle == d]) for d in unique_kk_values]
# q_low = np.percentile(a_err, 0.5)
# q_high = np.percentile(a_err, 97.5)
# error = (q_high - q_low) / 2
# plt.errorbar(kk_values, a_err, yerr=error, fmt='o', color='b', ecolor='r', capsize=5)
# plt.show()

# start_angle = np.array(start_angle)
# angle_err = np.array(angle_err)
# unique_start_angle = np.unique(start_angle)
# mean_angle_err = [np.mean(angle_err[start_angle == d]) for d in unique_start_angle]
# std_dev_angle_err = [np.std(angle_err[start_angle == d]) for d in unique_start_angle]
# print(std_dev_angle_err)
# plt.errorbar(unique_start_angle, mean_angle_err, yerr=std_dev_angle_err, fmt='o', color='b', ecolor='r', capsize=5)
# plt.xlabel('start_angle (°)', fontsize = 20)
# plt.ylabel('Angle error (°)', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.gca().xaxis.get_label().set_ha('center')  # 设置x轴标签居中对齐
# plt.gca().yaxis.get_label().set_ha('center')
# plt.grid()
# plt.show()

# d_values = np.array(d_values)
# b_err = np.array(b_err)
# unique_d_values, counts = np.unique(d_values, return_counts=True)
# mean_b_err = [np.mean(b_err[d_values == d]) for d in unique_d_values]
# std_dev_b_err = [np.std(b_err[d_values == d]) if count > 1 else 0 for d, count in zip(unique_d_values, counts)]
# # std_err_a_err = [np.std(a_err[kk_values == d]) / np.sqrt(count) if count > 1 else 0 for d, count in zip(unique_kk_values, counts)]
# # confidence_level = 0.95
# # conf_int_a_err = [t.interval(confidence_level, count - 1, loc=np.mean(a_err[kk_values == d]), scale=np.std(a_err[kk_values == d]) / np.sqrt(count)) if count > 1 else (0, 0) for d, count in zip(unique_kk_values, counts)]
# print("Counts:", counts)
# print("std_dev_a_err:", std_dev_b_err)

# plt.errorbar(unique_d_values, mean_b_err, yerr=std_dev_b_err, fmt='o', color='b', ecolor='r', capsize=5)
# plt.xlabel('start_angle (°)', fontsize=20)
# plt.ylabel('Angle error (°)', fontsize=20)
# plt.xticks(fontsize=20)  
# plt.yticks(fontsize=20)
# plt.gca().xaxis.get_label().set_ha('center')  # 设置x轴标签居中对齐
# plt.gca().yaxis.get_label().set_ha('center')
# plt.grid()
# plt.show()


# start_angle = np.array(start_angle)
# angle_err = np.array(angle_err)
# velocity_err = np.array(velocity_err)
# unique_start_angle = np.unique(start_angle)
# unique_start_angle , counts = np.unique(start_angle, return_counts=True)

# mean_angle_err = [np.mean(angle_err[start_angle == d]) for d in unique_start_angle]
# mean_velocity_err = [np.mean(velocity_err[start_angle == d]) for d in unique_start_angle]
# std_dev_angle_err = [np.std(angle_err[start_angle == d]) for d in unique_start_angle]
# std_dev_velocity_err = [np.std(velocity_err[start_angle == d]) for d in unique_start_angle]
# print("Counts:", counts)
# print("std_dev_a_err:", std_dev_angle_err)


# #计算平均值和原始数据的error bar图
# plt.errorbar(unique_start_angle, mean_angle_err, yerr=std_dev_angle_err, fmt='o', color='b', ecolor='r', capsize=5)
# # plt.plot(diameter, angle_err, 'bo')
# plt.xlabel('start_angle (°)', fontsize = 20)
# plt.ylabel('Angle error (°)', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.gca().xaxis.get_label().set_ha('center')  # 设置x轴标签居中对齐
# plt.gca().yaxis.get_label().set_ha('center')
# plt.grid()
# plt.show()



# plt.subplot(121)
# plt.plot(h_values, angle_err, 'bo')
# plt.xlabel('h(mm)', fontsize = 20)
# plt.ylabel('Angle error (°)', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.grid()
# plt.subplot(122)
# plt.plot(h_values, velocity_err, 'bo')
# plt.xlabel('h(mm)', fontsize = 20)
# plt.ylabel('Velocity error (m/s)', fontsize = 20)
# plt.xticks(fontsize = 20)  
# plt.yticks(fontsize = 20)
# plt.grid()
# plt.show()

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

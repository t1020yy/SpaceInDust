import pickle

import numpy as np
from matplotlib import pyplot as plt
from modeling import get_a_b_c

def calculate_relative_errors(values, values_1):
    return [(v - v_1) / v * 100 for v, v_1 in zip(values, values_1)]
def calculate_average_and_std_in_bins(values, bin_edges, errors):
    num_bins = len(bin_edges) - 1
    avg_errors = []
    std_errors = []

    for i in range(num_bins):
        indices_in_bin = np.where((values >= bin_edges[i]) & (values < bin_edges[i + 1]))[0]
        errors_bin = [errors[j] for j in indices_in_bin]

        avg_errors.append(abs(np.mean(errors_bin)))
        std_errors.append(np.std(errors_bin))

    return np.array(avg_errors), np.array(std_errors)

FILE_NAME = 'modeling_results_ 2023-11-15_19-36-29.pickle'

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

a_err_percent = calculate_relative_errors(a_values, a_1_values)
b_err_percent = calculate_relative_errors(b_values, b_1_values)
c_err_percent = calculate_relative_errors(c_values, c_1_values)

num_bins = 70  # You can adjust this value based on your needs

# Compute histogram of h_values
hist, bin_edges = np.histogram(kk_values, bins=num_bins)
avg_a_err_percent, std_a_err_percent = calculate_average_and_std_in_bins(kk_values, bin_edges, a_err_percent)
avg_b_err_percent, std_b_err_percent = calculate_average_and_std_in_bins(kk_values, bin_edges, b_err_percent)
avg_c_err_percent, std_c_err_percent = calculate_average_and_std_in_bins(kk_values, bin_edges, c_err_percent)

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
# Data for plotting
data = [
    {'label': 'a_err / a', 'avg': avg_a_err_percent, 'std': std_a_err_percent},
    {'label': 'b_err / b', 'avg': avg_b_err_percent, 'std': std_b_err_percent},
    {'label': 'c_err / c', 'avg': avg_c_err_percent, 'std': std_c_err_percent}
]

# Loop through the data and plot each subplot
for i, subplot_data in enumerate(data):
    ax = axs[i]
    ax.errorbar(bin_edges[:-1], subplot_data['avg'], yerr=subplot_data['std'], fmt='o', color='b', ecolor='r', capsize=5)
    ax.set_xlabel('kk', fontsize=16)
    ax.set_ylabel(f'{subplot_data["label"]} (%)', fontsize=16)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid()

# Adjust subplot layout to prevent overlap
plt.tight_layout()
# Display the plot
plt.show()

# # Create a single figure with a 1x1 grid of subplots
fig = plt.figure(figsize=(16, 5))
ax = fig.add_subplot(111)
# Plot each subplot within the larger subplot
for subplot_data in data:
    ax.errorbar(bin_edges[:-1], subplot_data['avg'], yerr=subplot_data['std'], fmt='o', label=f'{subplot_data["label"]} (%)', capsize=5)

# Set labels and title
ax.set_xlabel('kk', fontsize=16)
ax.set_ylabel('Error (%)', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid()
ax.legend()

# Display the plot
plt.show()

num_points_per_bin, _ = np.histogram(kk_values, bins=bin_edges)

# Plot the number of points per bin
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(bin_edges[:-1], num_points_per_bin, marker='o', linestyle='-', color='g')
ax.set_xlabel('kk', fontsize=16)
ax.set_ylabel('Number of Points', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid()

# Display the plots
plt.tight_layout()
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

# import numpy as np
# import matplotlib.pyplot as plt

# # 示例数据
# x_values = np.linspace(0, 10, 100)
# a_values = np.sin(x_values)
# b_values = 0.5 * x_values  # 请替换为你的实际数据

# # 计算 (a-b)/a
# relative_errors = (a_values - b_values) / a_values

# # 计算误差，这里简单使用标准差
# std_errors = np.random.uniform(0, 0.1, len(x_values))

# # 将误差转化为百分比
# std_errors_percent = (std_errors / a_values) * 100

# # 绘制曲线
# fig, ax1 = plt.subplots()

# # 绘制 (a-b)/a 的曲线
# ax1.plot(x_values, relative_errors, label='(a-b)/a', color='blue')
# ax1.set_xlabel('X-axis')
# ax1.set_ylabel('(a-b)/a', color='blue')
# ax1.tick_params(axis='y', labelcolor='blue')

# # 创建次坐标轴，用于绘制误差曲线
# ax2 = ax1.twinx()
# ax2.errorbar(x_values, relative_errors, yerr=std_errors_percent, fmt='o', label='Error Bar', color='red')

# # 设置次坐标轴的标签
# ax2.set_ylabel('Error (%)', color='red')
# ax2.tick_params(axis='y', labelcolor='red')

# # 添加图例
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')

# # 显示图形
# plt.show()

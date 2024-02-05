import pickle

import numpy as np
import sympy as sp
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

FILE_NAME = 'modeling_results_ 2023-11-24_02-19-53.pickle'

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
g=9.81
aa_values = []
bb_values = []
cc_values = []
hg_values = []

for i in range(len(parameters)):
    # if result[i] is not None:
    if i < len(result) and result[i] is not None:
        parameter = parameters[i]
        particle = result[i]
        a_1, b_1, c_1 = get_a_b_c(particle[1]*10**3, np.abs(particle[0]), parameter.x_start_trajectory, parameter.y_start_trajectory, g = 9.81*10**3)
        a_1_values.append(a_1)
        b_1_values.append(b_1)
        c_1_values.append(c_1)

        aa_values.append(particle[5])
        bb_values.append(particle[6])
        cc_values.append(particle[7])
        hg_values.append(particle[8])

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


aa, bb, cc, hg, x00, v00, alpha1 = sp.symbols('aa bb cc hg hx00 v00 alpha1')
x00 = ((-bb + ((bb ** 2 - 4 * aa* (cc - hg))**0.5)) / (2 * aa))
v00 = (g/(2*aa)+g*bb**2/(2*aa) + 2*bb*x00*g +2*x00**2*g*aa)**0.5
alpha1 = bb + x00 * 2 * aa
aa=np.mean(aa_values)
bb=np.mean(bb_values)
cc=np.mean(cc_values)
hg=np.mean(hg_values)
std_dev_aa = np.std(aa_values, ddof=0.5)
std_dev_bb = np.std(bb_values, ddof=0.5)
std_dev_cc = np.std(cc_values, ddof=0.5)
std_dev_hg = np.std(hg_values, ddof=0.5)

derivative_x00_aa = (-2.0*cc + 2.0*hg)/(2*aa*(-4*aa*(cc - hg) + bb**2)**0.5) - (-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/(2*aa**2)
derivative_x00_bb = (1.0*bb/(-4*aa*(cc - hg) + bb**2)**0.5 - 1)/(2*aa)
derivative_x00_cc = -1.0/(-4*aa*(cc - hg) + bb**2)**0.5
derivative_x00_hg = 1.0/(-4*aa*(cc - hg) + bb**2)**0.5
delta_a = std_dev_aa / np.sqrt(len(aa_values))
delta_b = std_dev_bb / np.sqrt(len(bb_values))
delta_c = std_dev_cc / np.sqrt(len(cc_values))
delta_hg = std_dev_hg / np.sqrt(len(hg_values))

measurement_uncertainty_x0 = (derivative_x00_aa**2*delta_a**2+ derivative_x00_bb**2*delta_b**2+derivative_x00_cc**2*delta_c**2+derivative_x00_hg**2*delta_hg**2)**0.5 
# v00 = (g/(2*aa)+g*bb**2/(2*aa) + 2*bb*x00*g +2*x00**2*g*aa)**0.5
v00 = (g/(2*aa)+g*bb**2/(2*aa) + 2*bb*((-bb + ((bb ** 2 - 4 * aa* (cc - hg))**0.5)) / (2 * aa))*g +2*((-bb + ((bb ** 2 - 4 * aa* (cc - hg))**0.5)) / (2 * aa))**2*g*aa)**0.5
derivative_v00_aa=3.132*(0.5*bb*(-2.0*cc + 2.0*hg)/(aa*(-4*aa*(cc - hg) + bb**2)**0.5) + 0.5*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)*(-2.0*cc + 2.0*hg)/(aa*(-4*aa*(cc - hg) + bb**2)**0.5) - 0.25*bb**2/aa**2 - 0.5*bb*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/aa**2 - 0.25*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)**2/aa**2 - 0.25/aa**2)/(0.5*bb**2/aa + bb*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/aa + 0.5*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)**2/aa + 0.5/aa)**0.5
derivative_v00_bb=3.132*(0.5*bb*(1.0*bb/(-4*aa*(cc - hg) + bb**2)**0.5 - 1)/aa + 0.5*bb/aa + 0.25*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)*(2.0*bb/(-4*aa*(cc - hg) + bb**2)**0.5 - 2)/aa + 0.5*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/aa)/(0.5*bb**2/aa + bb*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/aa + 0.5*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)**2/aa + 0.5/aa)**0.5
derivative_v00_cc=3.132*(-1.0*bb/(-4*aa*(cc - hg) + bb**2)**0.5 - 1.0*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/(-4*aa*(cc - hg) + bb**2)**0.5)/(0.5*bb**2/aa + bb*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/aa + 0.5*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)**2/aa + 0.5/aa)**0.5
derivative_v00_hg=3.132*(1.0*bb/(-4*aa*(cc - hg) + bb**2)**0.5 + 1.0*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/(-4*aa*(cc - hg) + bb**2)**0.5)/(0.5*bb**2/aa + bb*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)/aa + 0.5*(-bb + (-4*aa*(cc - hg) + bb**2)**0.5)**2/aa + 0.5/aa)**0.5
measurement_uncertainty_v0 = (derivative_v00_aa**2*delta_a**2+ derivative_v00_bb**2*delta_b**2+derivative_v00_cc**2*delta_c**2+derivative_v00_hg**2*delta_hg**2)**0.5 
# alpha1 = bb + x00 * 2 * aa
alpha1 = bb + ((-bb + ((bb ** 2 - 4 * aa* (cc - hg))**0.5)) / (2 * aa)) * 2 * aa
derivative_alpha1_aa=(-2.0*cc + 2.0*hg)/(-4*aa*(cc - hg) + bb**2)**0.5
derivative_alpha1_bb=1.0*bb/(-4*aa*(cc - hg) + bb**2)**0.5
derivative_alpha1_cc=-2.0*aa/(-4*aa*(cc - hg) + bb**2)**0.5
derivative_alpha1_hg=2.0*aa/(-4*aa*(cc - hg) + bb**2)**0.5
measurement_uncertainty_alpha1 = (derivative_alpha1_aa**2*delta_a**2+ derivative_alpha1_bb**2*delta_b**2+derivative_alpha1_cc**2*delta_c**2+derivative_alpha1_hg**2*delta_hg**2)**0.5 
# alpha1 = np.rad2deg(alpha1)

# print("测量值x00:", x00_values)

print(measurement_uncertainty_x0)
print(measurement_uncertainty_v0)
print(measurement_uncertainty_alpha1)


# a_err_percent = calculate_relative_errors(a_values, a_1_values)
# b_err_percent = calculate_relative_errors(b_values, b_1_values)
# c_err_percent = calculate_relative_errors(c_values, c_1_values)

# num_bins = 70 

# # Compute histogram of h_values
# hist, bin_edges = np.histogram(kk_values, bins=num_bins)
# avg_a_err_percent, std_a_err_percent = calculate_average_and_std_in_bins(kk_values, bin_edges, a_err_percent)
# avg_b_err_percent, std_b_err_percent = calculate_average_and_std_in_bins(kk_values, bin_edges, b_err_percent)
# avg_c_err_percent, std_c_err_percent = calculate_average_and_std_in_bins(kk_values, bin_edges, c_err_percent)

# fig, axs = plt.subplots(1, 3, figsize=(16, 5))
# # Data for plotting
# data = [
#     {'label': 'a_err / a', 'avg': avg_a_err_percent, 'std': std_a_err_percent},
#     {'label': 'b_err / b', 'avg': avg_b_err_percent, 'std': std_b_err_percent},
#     {'label': 'c_err / c', 'avg': avg_c_err_percent, 'std': std_c_err_percent}
# ]

# # Loop through the data and plot each subplot
# for i, subplot_data in enumerate(data):
#     ax = axs[i]
#     ax.errorbar(bin_edges[:-1], subplot_data['avg'], yerr=subplot_data['std'], fmt='o', color='b', ecolor='r', capsize=5)
#     ax.set_xlabel('kk', fontsize=16)
#     ax.set_ylabel(f'{subplot_data["label"]} (%)', fontsize=16)
#     ax.tick_params(axis='both', labelsize=16)
#     ax.grid()

# # Adjust subplot layout to prevent overlap
# plt.tight_layout()
# # Display the plot
# plt.show()

# # # Create a single figure with a 1x1 grid of subplots
# fig = plt.figure(figsize=(16, 5))
# ax = fig.add_subplot(111)
# # Plot each subplot within the larger subplot
# for subplot_data in data:
#     ax.errorbar(bin_edges[:-1], subplot_data['avg'], yerr=subplot_data['std'], fmt='o', label=f'{subplot_data["label"]} (%)', capsize=5)

# # Set labels and title
# ax.set_xlabel('kk', fontsize=16)
# ax.set_ylabel('Error (%)', fontsize=16)
# ax.tick_params(axis='both', labelsize=16)
# ax.grid()
# ax.legend()

# # Display the plot
# plt.show()

# num_points_per_bin, _ = np.histogram(kk_values, bins=bin_edges)

# # Plot the number of points per bin
# fig, ax = plt.subplots(figsize=(8, 5))
# ax.plot(bin_edges[:-1], num_points_per_bin, marker='o', linestyle='-', color='g')
# ax.set_xlabel('kk', fontsize=16)
# ax.set_ylabel('Number of Points', fontsize=16)
# ax.tick_params(axis='both', labelsize=16)
# ax.grid()

# # Display the plots
# plt.tight_layout()
# plt.show()



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

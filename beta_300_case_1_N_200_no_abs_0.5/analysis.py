import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import torch
import torchvision
import linpdeconfig
from torch.autograd import Variable
import matplotlib.pyplot as plt
from pltutils import *
# import seaborn as sns
import re



def plot_fixed_time_step_different_beta_0_1(real_0_data, real_1_data, real_0_step, real_1_step, i, time, N, example_index, name):
    time_steps_0 = [0, 10, 21, 31, 42, 52, 63, 73, 84, 94]
    time_steps_1 = [0, 64, 128, 192, 255, 319, 383, 447, 511, 575]
    # for i in range(1, len(time_steps_0)):
    fix_timestep_0_data = real_0_data[time_steps_0[i], example_index, :]
    fix_timestep_1_data = real_1_data[time_steps_1[i], example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/N   # 0.05
    for i in range(N):
        x_label.append(i * dx)
    plot_0, = plt.plot(x_label, fix_timestep_0_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_1, = plt.plot(x_label, fix_timestep_1_data, label='prediction', color='blue', linestyle='-', linewidth=linewith)


    # plt.title(label='X', fontsize=title_fontsize)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_0, plot_1,],
               labels=['0', '1.0',],
               loc="upper right", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    # save_dir = 'figures/' + name + '_' + 'time_' + str(time)
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # # file_name = '/' + name + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    # file_name = '/' + name + '_' + str(example_index) + '_' + str(time) + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def area(p):
    return np.abs(np.cross(p, np.roll(p, 1, axis=0)).sum()) / 2


def plot_fixed_time_difference():
    N = 200
    time_step = 8  # 3, 5
    example_index = 0
    test_data_name = 'beta_300_case_1_N_200_0.5'
    data_0_file = 'data/beta_300_case_1_N_200_0.5_without_diff_U.npy'
    data_10_file = 'data/beta_300_case_1_N_200_0.5_U.npy'
    time_steps_0 = [0, 11, 23, 34, 45, 57, 68, 79, 91, 102]
    time_steps_10 = [0, 65, 129, 194, 259, 323, 388, 452, 517, 582]
    real_0_step = time_steps_0[time_step]
    real_10_step = time_steps_10[time_step]
    real_0_data = np.load(data_0_file)[real_0_step, example_index, :]
    real_10_data = np.load(data_10_file)[real_10_step, example_index, :]

    linewith = 3.0
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 35

    x_label = []
    dx = 10 / N  # 0.05
    for i in range(N):
        x_label.append((i+0.5) * dx)

    x1 = x_label
    x2 = x_label
    y1 = real_0_data
    y2 = real_10_data

    xy0 = np.array([(x, y) for x, y in zip(x1, y1)])
    xy1 = np.array([(x, y) for x, y in zip(x2, y2)])
    p = np.r_[xy0, xy1[::-1]]
    result1 = area(p)

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    plot_1, = plt.plot(*xy0.T, 'b-', label=r'$\alpha=0.0$', linewidth=linewith)
    plot_2, = plt.plot(*xy1.T, 'r-', label=r'$\alpha=0.5$', linewidth=linewith)
    p = np.r_[xy0, xy1[::-1]]
    plt.fill(*p.T, alpha=.2)

    # 添加水平方向的线
    plt.hlines(0.25, xmin=0.0, xmax=10, ls="--", color='black')
    plt.hlines(0.5, xmin=0.0, xmax=10, ls="--", color='black')
    plt.hlines(0.75, xmin=0.0, xmax=10, ls="--", color='black')

    # plt.title(label='X', fontsize=title_fontsize)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    y_ticks = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25,  0.3, 0.35, 0.4, 0.45, 0.5, 0.55,
               0.6, 0.65,  0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(y_ticks, fontsize=ticks_fontsize)
    plt.xlim(0.0, 10.0)
    # plt.ylim(0.0, 1.0)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2],
               labels=[r'$\alpha=0.0$', r'$\alpha=0.5$'],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.show()
    save_dir = 'figures/' + test_data_name + '_obs'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/exam_' + str(example_index) + '_time_' + str(time_step * 0.1) + '_a1' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


if __name__ == "__main__":
    N = 200
    test_data_name = 'beta_300_case_1_N_200_0.5'
    data_0_file = 'data/beta_300_case_1_N_200_0.5_without_diff_U.npy'
    data_10_file = 'data/beta_300_case_1_N_200_0.5_U.npy'
    real_0_data = np.load(data_0_file)
    real_10_data = np.load(data_10_file)
    time_steps_0 = [0, 11, 23, 34, 45, 57, 68, 79, 91, 102]
    time_steps_10 = [0, 65, 129, 194, 259, 323, 388, 452, 517, 582]

    # 画出a(u)在各个时刻会解的影响效果
    # plot_fixed_time_difference()

    # example_index = 6
    # time_step = 9
    diff_u_75 = 0.0
    diff_u_50_75 = 0.0
    diff_u_25_50 = 0.0
    diff_u_25 = 0.0
    # range(0, 6)
    # range(1, 10)
    for example_index in range(0, 6):
        print("example is %f" % (example_index))
        for time_step in range(1, 10):
            time = time_step * 0.1
            print("time is %f" % (time))

            real_0_step = time_steps_0[time_step]
            real_10_step = time_steps_10[time_step]
            # plot_fixed_time_step_different_beta_0_1(real_0_data, real_10_data, real_0_step, real_10_step, time_step, time, N, example_index, name=test_data_name)


            #u >= 0.75
            # 比较每一段u上的误差大小
            real_0_data = np.load(data_0_file)[real_0_step, example_index, :]
            real_10_data = np.load(data_10_file)[real_10_step, example_index, :]
            where_res_0 = np.where(real_0_data >= 0.75)
            where_res_10 = np.where(real_10_data >= 0.75)
            res_0 = real_0_data[where_res_0]
            res_10 = real_10_data[where_res_10]

            x1 = where_res_0[0] * 0.05
            x2 = where_res_10[0] * 0.05
            y1 = res_0
            y2 = res_10

            xy0 = np.array([(x, y) for x, y in zip(x1, y1)])

            if len(res_0) == 0 and len(res_10) == 0:
                result1 = 0.0
                print("ares is %f" % (result1))
                diff_u_75 += result1
            elif len(res_0) != 0 and len(res_10) == 0:
                y2 = np.zeros_like(y1) + 0.75
                xy1 = np.array([(x, y) for x, y in zip(x1, y2)])
                p = np.r_[xy0, xy1[::-1]]
                result1 = area(p)
                print("ares is %f" % (result1))
                diff_u_75 += result1
            else:
                xy1 = np.array([(x, y) for x, y in zip(x2, y2)])
                p = np.r_[xy0, xy1[::-1]]
                result1 = area(p)
                print("ares is %f" % (result1))
                diff_u_75 += result1

            # plt.plot(*xy0.T, 'b-')
            # plt.plot(*xy1.T, 'r-')
            # p = np.r_[xy0, xy1[::-1]]
            # plt.fill(*p.T, alpha=.2)
            # plt.show()

            # u >= 0.5 and u  < 0.75
            real_0_data = np.load(data_0_file)[real_0_step, example_index, :]
            real_10_data = np.load(data_10_file)[real_10_step, example_index, :]
            where_res_0 = np.where((real_0_data >= 0.5) & (real_0_data < 0.75))
            where_res_10 = np.where((real_10_data >= 0.5) & (real_10_data < 0.75))
            res_0 = real_0_data[where_res_0]
            res_10 = real_10_data[where_res_10]

            x1 = where_res_0[0] * 0.05
            x2 = where_res_10[0] * 0.05
            y1 = res_0
            y2 = res_10

            xy0 = np.array([(x, y) for x, y in zip(x1, y1)])
            xy1 = np.array([(x, y) for x, y in zip(x2, y2)])

            if len(res_0) == 0 and len(res_10) == 0:
                result2 = 0.0
                print("ares is %f" % (result1))
                diff_u_50_75 += result2
            elif len(res_0) != 0 and len(res_10) == 0:
                y2 = np.zeros_like(y1) + 0.5
                xy1 = np.array([(x, y) for x, y in zip(x1, y2)])
                p = np.r_[xy0, xy1[::-1]]
                result2 = area(p)
                print("ares is %f" % (result2))
                diff_u_50_75 += result2
            elif len(res_0) == 0 and len(res_10) != 0:
                y1 = np.zeros_like(y2) + 0.5
                xy0 = np.array([(x, y) for x, y in zip(x2, y1)])
                p = np.r_[xy0, xy1[::-1]]
                result2 = area(p)
                print("ares is %f" % (result2))
                diff_u_50_75 += result2
            else:
                xy1 = np.array([(x, y) for x, y in zip(x2, y2)])
                p = np.r_[xy0, xy1[::-1]]
                result2 = area(p)
                print("ares is %f" % (result2))
                diff_u_50_75 += result2

            # plt.plot(*xy0.T, 'b-')
            # plt.plot(*xy1.T, 'r-')
            # p = np.r_[xy0, xy1[::-1]]
            # plt.fill(*p.T, alpha=.2)
            # plt.show()

            # u >= 0.25 and u  < 0.5
            real_0_data = np.load(data_0_file)[real_0_step, example_index, :]
            real_10_data = np.load(data_10_file)[real_10_step, example_index, :]
            where_res_0 = np.where((real_0_data >= 0.25) & (real_0_data < 0.5))
            where_res_10 = np.where((real_10_data >= 0.25) & (real_10_data < 0.5))
            res_0 = real_0_data[where_res_0]
            res_10 = real_10_data[where_res_10]

            x1 = where_res_0[0] * 0.05
            x2 = where_res_10[0] * 0.05
            y1 = res_0
            y2 = res_10

            xy0 = np.array([(x, y) for x, y in zip(x1, y1)])
            xy1 = np.array([(x, y) for x, y in zip(x2, y2)])

            p = np.r_[xy0, xy1[::-1]]
            result3 = area(p)
            print("ares is %f" % (result3))
            diff_u_25_50 += result3

            # plt.plot(*xy0.T, 'b-')
            # plt.plot(*xy1.T, 'r-')
            # p = np.r_[xy0, xy1[::-1]]
            # plt.fill(*p.T, alpha=.2)
            # plt.show()

            # u < 0.25
            real_0_data = np.load(data_0_file)[real_0_step, example_index, :]
            real_10_data = np.load(data_10_file)[real_10_step, example_index, :]
            where_res_0 = np.where((real_0_data < 0.25))
            where_res_10 = np.where((real_10_data < 0.25))
            res_0 = real_0_data[where_res_0]
            res_10 = real_10_data[where_res_10]

            x1 = where_res_0[0] * 0.05
            x2 = where_res_10[0] * 0.05
            y1 = res_0
            y2 = res_10

            if (len(y1) == 0 and len(y2) == 0):
                result4 = 0.0
                print("ares is %f" % (result4))
            else:
                xy0 = np.array([(x, y) for x, y in zip(x1, y1)])
                xy1 = np.array([(x, y) for x, y in zip(x2, y2)])

                p = np.r_[xy0, xy1[::-1]]
                result4 = area(p)
                print("ares is %f" % (result4))

                # plt.plot(*xy0.T, 'b-')
                # plt.plot(*xy1.T, 'r-')
                # p = np.r_[xy0, xy1[::-1]]
                # plt.fill(*p.T, alpha=.2)
                # plt.show()
            diff_u_25 += result4
            print("#############################################")

    print((diff_u_75, diff_u_50_75, diff_u_25_50, diff_u_25))










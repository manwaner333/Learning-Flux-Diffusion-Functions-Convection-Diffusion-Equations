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

# sns.set_theme()

# def plot_real_and_predict_data(real_data, predict_data, example_index, name=''):
#     fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)
#     g1 = sns.heatmap(real_data[:, example_index, :].T, cmap="YlGnBu", cbar=True, ax=axes[0])
#     g1.set_ylabel('cell')
#     g1.set_xlabel('time_step')
#     g2 = sns.heatmap(predict_data[:, example_index, :].T, cmap="YlGnBu", cbar=True, ax=axes[1])
#     g2.set_ylabel('cell')
#     g2.set_xlabel('time_step')
#     g3 = sns.heatmap(real_data[:, example_index, :].T - predict_data[:, example_index, :].T
#                      , cmap="YlGnBu", cbar=True, ax=axes[2])
#     g3.set_ylabel('cell')
#     g3.set_xlabel('time_step')
#     plt.show()
#
#     save_dir = 'figures/' + name
#     if not os.path.isdir(save_dir):
#         os.mkdir(save_dir)
#     file_name = '/real_predict_loss_example' + '_' + str(example_index) + '_' + '.pdf'
#     fig.savefig(save_dir + file_name)


# def plot_real_and_predict_data(real_data, predict_data, example_index, name=''):
#     fig, axes = plt.subplots(1, 2, figsize=(50, 16), sharey=True)
#     linewith = 10
#     linewith_frame = 4
#     title_fontsize = 60
#     label_fontsize = 50
#     ticks_fontsize = 50
#     cbar_size = 50
#     g1 = sns.heatmap(np.flip(real_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, annot_kws={"size":30}, ax=axes[0])
#     g1.set_ylabel('T', fontsize=label_fontsize)
#     g1.set_xlabel('X', fontsize=label_fontsize)
#     g1.set_xticklabels([])
#     g1.set_yticklabels([])
#     # plt.xticks(np.arange(0, 2, step=0.2),list('abcdefghigk'),rotation=45)
#     g1.set_title("exact u(x, t)", fontsize=title_fontsize)
#     cax1 = plt.gcf().axes[-1]
#     cax1.tick_params(labelsize=cbar_size)
#     cax1.spines['top'].set_linewidth(linewith_frame)
#     cax1.spines['bottom'].set_linewidth(linewith_frame)
#     cax1.spines['right'].set_linewidth(linewith_frame)
#     cax1.spines['left'].set_linewidth(linewith_frame)
#
#     g2 = sns.heatmap(np.flip(predict_data[:, example_index, :], 0), cmap="YlGnBu", cbar=True, ax=axes[1])
#     g2.set_ylabel('T', fontsize=label_fontsize)
#     g2.set_xlabel('X', fontsize=label_fontsize)
#     g2.set_xticklabels([])
#     g2.set_yticklabels([])
#     g2.set_title("prediction u(x, t)", fontsize=title_fontsize)
#     cax2 = plt.gcf().axes[-1]
#     cax2.tick_params(labelsize=cbar_size)
#     cax2.spines['top'].set_linewidth(linewith_frame)
#     cax2.spines['bottom'].set_linewidth(linewith_frame)
#     cax2.spines['right'].set_linewidth(linewith_frame)
#     cax2.spines['left'].set_linewidth(linewith_frame)
#
#
#
#     # g3 = sns.heatmap(np.flip(real_data[:, example_index, :], 0) - np.flip(predict_data[:, example_index, :], 0)
#     #                  , cmap="YlGnBu", cbar=True, ax=axes[2])
#     # g3.set_ylabel('T', fontsize=label_fontsize)
#     # g3.set_xlabel('X', fontsize=label_fontsize)
#     # g3.set_xticklabels([])
#     # g3.set_yticklabels([])
#     # g3.set_title("error u(x, t)", fontsize=title_fontsize)
#     # cax = plt.gcf().axes[-1]
#     # cax.tick_params(labelsize=cbar_size)
#
#     plt.show()
#
#     # save_dir = 'figures/' + name
#     # if not os.path.isdir(save_dir):
#     #     os.mkdir(save_dir)
#     # file_name = '/real_predict_data_example' + '_' + str(example_index) + '_all_time' + '.pdf'
#     # fig.savefig(save_dir + file_name)


def plot_fixed_time_step(real_data, predict_data, time_steps_real, time_steps_predict, time, N, example_index, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/N   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data, label='prediction', color='blue', linestyle='--', linewidth=linewith)
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
    plt.legend(handles=[plot_1, plot_2],
               labels=['exact', 'prediction'],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '_' + 'time_' + str(time)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    # file_name = '/' + name + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    file_name = '/' + name + '_' + str(example_index) + '_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def f_real_function(u):
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * 300 * np.power(u, 2) * ((3/4) - 2*u + (3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f

def f_predict_function(u):
    f = (66421.4118566256)*u**28+(65178.044212212575)*u**21+(-64928.00908003173)*u**20+(-64398.06105846598)*u**29+(-60602.4191500477)*u**27+(56760.75809958727)*u**30+(55671.44067967981)*u**19+(-54388.02726880888)*u**22+(-45980.79896159293)*u**31+(45704.91078913276)*u**26+(-40861.2229857805)*u**18+(34332.43573168292)*u**32+(33264.74442494944)*u**23+(24299.791318145457)*u**17+(-23606.170891302005)*u**33+(-22476.31580984383)*u**25+(14912.013379635293)*u**34+(-9573.570446382571)*u**16+(-8633.702834344189)*u**35+(-5694.260523594717)*u**24+(4923.548038841936)*u**14+(-4588.843188306411)*u**13+(4572.63086253846)*u**36+(-2212.330074348854)*u**37+(2032.5393450176482)*u**12+(-1221.404279605525)*u**10+(989.6215688975959)*u**9+(976.9412183532254)*u**38+(-540.4758581062612)*u**15+(-409.7767846914912)*u**8+(-393.54306194906854)*u**39+(292.1717899801514)*u**11+(144.57022126483807)*u**40+(55.859372817804164)*u**6+(52.99721245715598)*u**4+(-51.53008652487216)*u**3+(-48.419237067933814)*u**41+(-44.80869219164241)*u**5+(37.34986422496874)*u**7+(17.720010256037302)*u**2+(14.78045857072267)*u**42+(-4.110749352813037)*u**43+(1.6841825910039323)*u+(1.0410565592995358)*u**44+(-0.26396004342830265)*1+(-0.23988441457475992)*u**45+(0.050236807210104255)*u**46+(-0.009547001572679107)*u**47+(0.0016428727580458458)*u**48+(-0.0002552117791876214)*u**49+(3.5629812390841766e-05)*u**50+(-4.440846004454221e-06)*u**51
    return f

def a_real_function(u):
    if u < 0.5:
        f = 0.0
    else:
        f = 1.0
        # f = (u - 0.5) * u
    return f
# def a_predict_function(u):
#     if u <= 0.5:
#         y = (-0.656464743609309)*1+(-0.19078714955844395)*u
#     else:
#         y = (1.1412332456160317)*1+(-0.19258733840113776)*u
#     if y <= 0.0:
#         return 0.0
#     else:
#         return y
def a_predict_function(u):
    if u <= 0.25:
        y = (0.12392774057257737)*u+(-0.012357270382967828)*1
    elif u > 0.25 and u < 0.5:
        y = (-0.09383728698826124)*u+(0.005656356935054112)*1
    elif u >= 0.5 and u < 0.75:
        y = (1.1672482346820114)*u+(0.3317831322851484)*1
    else:
        y = (-0.6147395143123183)*u+(-0.25997089236418663)*1
    return np.abs(y)



def A_real_function():
    N_a = 400
    u_a_np = np.zeros(N_a + 1, dtype=float)
    u_a_0 = 0.0
    u_a_np[0] = u_a_0
    for i in range(1, N_a + 1):
        u_a_np[i] = u_a_0 + (1.0 / N_a) * i
    a_real = np.array([a_real_function(ele) for ele in u_a_np])
    A_a_real = np.zeros(N_a + 1, dtype=float)
    for index in range(1, N_a + 2):
        A_a_real[index - 1] = np.trapz(a_real[0:index], u_a_np[0:index])
    return A_a_real

def A_predict_function():
    N_a = 400
    u_a_np = np.zeros(N_a + 1, dtype=float)
    u_a_0 = 0.0
    u_a_np[0] = u_a_0
    for i in range(1, N_a + 1):
        u_a_np[i] = u_a_0 + (1.0 / N_a) * i
    a_predict = np.array([a_predict_function(ele) for ele in u_a_np])
    A_a_predict = np.zeros(N_a + 1, dtype=float)
    for index in range(1, N_a + 2):
        A_a_predict[index - 1] = np.trapz(a_predict[0:index], u_a_np[0:index])
    return A_a_predict

def plot_real_and_predict_function(name='name'):
    x = []
    N = 100
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    f_real = np.array([f_real_function(ele) for ele in x])
    f_predict = np.array([f_predict_function(ele) for ele in x])

    linewith = 3
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    # plt.plot()
    plot_1, = plt.plot(x, f_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, f_predict - (f_predict[0] - f_real[0]), label='fix', color='blue', linestyle='-', linewidth=linewith)

    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('f(u)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_3],  # ,  plot_2
               labels=['exact', 'revised prediction'],  # 'prediction',
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    # ax1.set_facecolor('w')
    plt.show()
    #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_f' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_real_and_predict_a_function(name='name'):
    x = []
    N = 100
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    f_real = np.array([a_real_function(ele) for ele in x])
    f_predict = np.array([a_predict_function(ele) for ele in x])

    linewith = 3.0
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, f_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, f_predict, label='fix', color='blue', linestyle='-', linewidth=linewith)

    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('a(u)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_3],  # ,  plot_2
               labels=['exact', 'prediction'],  # 'prediction',
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    # ax1.set_facecolor('w')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_a' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_real_and_predict_A_function(name='name'):
    A_real = A_real_function()
    A_predict = A_predict_function()
    x = []
    N = 400
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    linewith = 3.0
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, A_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, A_predict, label='fix', color='blue', linestyle='-', linewidth=linewith)

    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('A(u)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_3],  # ,  plot_2
               labels=['exact', 'prediction'],  # 'prediction',
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    # ax1.set_facecolor('w')
    plt.show()
    #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_A_u' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_observation_distribution(time_points, name):
    all_data = np.load('data/' + name + '_U.npy')
    time_points = time_points
    data = all_data[time_points, :, :].flatten()
    # new_data = []
    # for ele in data:
    #     if ele != 0.0 and ele != 1.0 and ele != 0.8 and ele != 0.6 and ele != 0.4 and ele != 0.3 and ele != 0.7:
    #         new_data.append(ele)
    new_data = data
    weights = [1./len(new_data)] * len(new_data)
    label_fontsize = 50
    ticks_fontsize = 50
    linewith_frame = 4
    fig = plt.figure(figsize=(40, 20))
    # plt.hist(new_data, weights=weights, bins=50)
    plt.hist(new_data, bins=50)
    plt.ylim(0, 1000)
    plt.xlabel('u', fontsize=label_fontsize)
    plt.ylabel('proportion', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    # major_ticks_top = np.linspace(0, 1, 21)
    # ax1.set_yticks(major_ticks_top)
    plt.grid(linewidth=0.2)
    plt.show()
    # #  保存
    # save_dir = 'figures/' + name + '_observation_distribution'
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # file_name = '/' + 'observation_distribution_all' + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_fixed_time_step_add_noise(real_data, real_no_noise_data, predict_data, time_steps_real, time_steps_predict, time, example_index,cell_numbers, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_real_no_noise_data = real_no_noise_data[time_steps_real, example_index, :]
    fix_timestep_predict_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_no_noise_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2, = plt.plot(x_label, fix_timestep_predict_data, label='prediction', color='blue', linestyle='--', linewidth=linewith)
    plot_3 = plt.scatter(x_label, fix_timestep_real_data, c="black")
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
    plt.legend(handles=[plot_1, plot_2, plot_3],
               labels=['exact', 'prediction', 'noise'],
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.show()

    save_dir = 'figures/' + name + '_' +'time_' + str(time)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_predict_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_init_states(data, example_index, name):
    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    init_states = data[0]
    init_state = init_states[example_index]
    x_label = []
    dx = 10/400   # 0.05
    for i in range(400):
        x_label.append(i * dx)
    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    plot_1, = plt.plot(x_label, init_state, label='observation', color='blue', linestyle='-', linewidth=linewith)
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.ylim(-0.1, 1.1)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_initial_states'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'initial_state_data' + '_' + 'example' + '_' + str(example_index) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


# 画出真实和噪声数据
def plot_fixed_time_step_real_noise_data(real_data, predict_data, time_steps_real, time_steps_predict, time, example_index, name):

    fix_timestep_real_data = real_data[time_steps_real, example_index, :]
    fix_timestep_noise_data = predict_data[time_steps_predict, example_index, :]

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(30, 20))
    plt.plot()
    x_label = []
    dx = 10/200   # 0.05
    for i in range(len(fix_timestep_real_data)):
        x_label.append(i * dx)
    plot_1, = plt.plot(x_label, fix_timestep_real_data, label='observation', color='red', linestyle='-', linewidth=linewith)
    plot_2 = plt.scatter(x_label, fix_timestep_noise_data, c="black")
    plt.xlabel('x', fontsize=label_fontsize)
    plt.ylabel('u(x)', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)
    plt.legend(handles=[plot_1, plot_2],
               labels=['clean', 'noise'],
               loc="upper right", fontsize=50, frameon=True, edgecolor='green')
    # plt.show()

    save_dir = 'figures/' + name + '_' + 'exact_noise'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'fixed_time_step_real_noise_data' + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def plot_loss_function_all(name):

    f = open("checkpoint/" + name + "/" + "loss.txt", encoding="utf8")
    line = f.readline()
    value_list = []
    while line:
        if "data loss0" in line and "nan" not in line:
            # print(line)
            number = re.findall(r'\d+(?:\.\d+)?', line)
            # print(number)
            value_list.append(float(number[1]))
        line = f.readline()
    value_list.remove(0.422315)
    x = np.arange(0, len(value_list))
    linewith = 5
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    plot_1, = plt.plot(x, value_list, label='loss', color='red', linestyle='-', linewidth=linewith)
    plt.xlabel('Iterations', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)

    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_loss'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'loss_function_all' + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_loss_function_sub(name, index, limit):

    f = open("checkpoint/" + name + "/" + "loss.txt", encoding="utf8")
    line = f.readline()
    value_list = []
    while line:
        if "data loss0" in line and "nan" not in line:
            # print(line)
            number = re.findall(r'\d+(?:\.\d+)?', line)
            # print(number)
            value_list.append(float(number[1]))
        line = f.readline()
    linewith = 2
    linewith_frame = 1
    label_fontsize = 10
    ticks_fontsize = 10

    new_value_list = []
    for i in range(len(value_list)):
        if i >= index:
            new_value_list.append(value_list[i])
    x = np.arange(0, len(new_value_list))
    fig = plt.figure(figsize=(8, 5))
    plot_1, = plt.plot(x, new_value_list, label='loss', color='green', linestyle='-', linewidth=linewith)
    # plt.xlabel('Iterations', fontsize=label_fontsize)
    plt.ylabel('Loss', fontsize=label_fontsize)
    plt.xticks(fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.xlim(0, len(new_value_list))
    plt.ylim(0.0, limit)
    ax1 = plt.gca()
    ax1.spines['top'].set_linewidth(linewith_frame)
    ax1.spines['bottom'].set_linewidth(linewith_frame)
    ax1.spines['right'].set_linewidth(linewith_frame)
    ax1.spines['left'].set_linewidth(linewith_frame)

    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_loss'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + 'loss_function_sub' + '.png'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)







if __name__ == "__main__":
    N = 200
    test_data_name = 'beta_300_case_1_N_200_0.5_obs_3'
    real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    predict_data_file = 'data/' + 'predict_' + test_data_name + '_U' + '.npy'

    # real_time_step = 1293
    # predict_time_step = 1500  # 1480
    # time = 2
    # example_index = 0
    # real_data = np.load(real_data_file)
    # predict_data = np.load(predict_data_file)

    # # 真实和预测以及误差情况
    # plot_real_and_predict_data(real_data, predict_data, example_index, name=test_data_name)

    # 固定时间下真实和预测情况
    # (不含有噪声)
    # plot_fixed_time_step(real_data, predict_data, real_time_step, predict_time_step, time, N, example_index, name=test_data_name)
    #（含有噪声）
    # cell_numbers = N
    # real_no_noise_data = np.load('data/N_400_example_2_dt_0.1_layer_10_beta_0.5_no_alpha_U' + '.npy')
    # plot_fixed_time_step_add_noise(real_data, real_no_noise_data, predict_data, real_time_step, predict_time_step, time, example_index, cell_numbers, name=test_data_name)

    # f(u)函数的差别
    plot_real_and_predict_function(name=test_data_name)
    plot_real_and_predict_a_function(name=test_data_name)
    plot_real_and_predict_A_function(name=test_data_name)

    # 画出观测数据得分布图
    # time_point_list = [0, 44, 88, 132, 176, 220, 264, 308, 352, 396]
    # plot_observation_distribution(time_points=time_point_list, name=test_data_name)

    # 画出噪音数据下的初始状态
    # example_index = 0
    # plot_init_states(real_data, example_index, test_data_name)



    # 比较真实数据和加了误差的数据
    # time_steps_real = 101   # 34(03), 67(06), 101(09)
    # time_steps_predict = time_steps_real
    # time = 0.9
    # example_index = 0
    # plot_fixed_time_step_real_noise_data(real_no_noise_data, real_data, time_steps_real, time_steps_predict, time, example_index, name=test_data_name)



    # 画出损失函数
    # plot_loss_function_all(name=test_data_name)
    # index = 168
    # limit = 0.0001
    # plot_loss_function_sub(name=test_data_name, index=index, limit=limit)








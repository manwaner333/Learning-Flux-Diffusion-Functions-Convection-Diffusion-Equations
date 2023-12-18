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

    # save_dir = 'figures/' + name + '_' + 'time_' + str(time)
    # if not os.path.isdir(save_dir):
    #     os.mkdir(save_dir)
    # file_name = '/' + name + '_' + 'example' + '_' + str(example_index) + '_time_' + str(time) + '.pdf'
    # fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def f_real_function(u):
    f = 0.5 * u * (3 - np.power(u, 2)) + (1/12) * 300 * np.power(u, 2) * ((3/4) - 2*u + (3/2)*np.power(u, 2) - (1/4)*np.power(u, 4))
    return f

def f_predict_function(u):
    # f = (1.6078884582912254)*u+(-0.9122358972596538)*1+(-0.4865418572526484)*u**3+(-0.13751993663563336)*u**2+(0.027555871969427246)*u**5+(-0.005948682382410473)*u**4+(-0.001006452518445636)*u**6+(-0.00014497184663870774)*u**7+(-2.7155610338378887e-06)*u**8
    # f = (57.29138871259054)*u**10+(-43.332218342966456)*u**8+(-40.02341002232253)*u**9+(32.53537284980668)*u**6+(27.64250263737908)*u**7+(19.697285545199566)*u**15+(-18.829170200691465)*u**4+(-17.26521898903823)*u**16+(10.038360744193891)*u**17+(-7.170055655781945)*u**12+(-6.723530855147927)*u**13+(-6.371583371831224)*u**18+(-5.125293028924684)*u**14+(-5.042133220382549)*u**5+(4.333607524324953)*u**2+(4.111587214264377)*u**19+(-3.2971902106260766)*u**11+(-2.972119828113396)*u**3+(2.9462489529172395)*u+(-1.9069732149976453)*u**20+(1.5181650084128837)*1+(0.4838973854405513)*u**21+(-0.06551625802911089)*u**23+(0.028683862377981314)*u**24+(0.013316927318345549)*u**22+(-0.006575149822360054)*u**25+(0.0005272009015388277)*u**26+(0.0001992254313235825)*u**27+(-9.209733024022466e-05)*u**28+(2.0434442735403258e-05)*u**29+(-2.8513754645175264e-06)*u**30
    f = (-25972.371598619586)*u**25+(25840.812343138885)*u**24+(-19884.140782744547)*u**20+(17085.345069783354)*u**26+(16357.674571102249)*u**21+(-14825.732069733394)*u**23+(13501.289357667172)*u**19+(8645.249702576904)*u**15+(-7236.374039981482)*u**14+(7102.099097157273)*u**29+(-5493.341258068011)*u**16+(-5219.3700543720215)*u**30+(-5162.229419639262)*u**27+(-5121.4548944559865)*u**18+(5007.950820777607)*u**34+(-4897.029399809083)*u**33+(-3982.450361365026)*u**35+(-3865.0478416226433)*u**28+(2894.029956532971)*u**32+(2665.6699177649016)*u**13+(2649.0277282712677)*u**36+(2356.5193091271185)*u**17+(-2334.901552054318)*u**22+(-1522.570290185906)*u**37+(965.964494891417)*u**31+(-920.5920884029395)*u**11+(769.6499860336208)*u**38+(670.4323238231805)*u**12+(650.4757696247805)*u**9+(-345.8635858606117)*u**39+(250.29415112615064)*u**6+(-231.45564683774361)*u**7+(-217.14302316246128)*u**8+(-204.15745327222774)*u**10+(139.140488682436)*u**40+(-90.73948481593175)*u**5+(-50.34902781777148)*u**41+(-42.633920146288844)*u**3+(38.2010108054874)*u**4+(16.868944239777186)*u**2+(16.44035687554242)*u**42+(-4.8543850866942835)*u**43+(1.6251119401045502)*u+(1.2978165520437026)*u**44+(-0.3699464487222815)*1+(-0.31433603585388764)*u**45+(0.06896670683361808)*u**46+(-0.013696797405731096)*u**47+(0.0024584198521216682)*u**48+(-0.00039775227157072264)*u**49+(5.7768516450493116e-05)*u**50+(-7.4838375300789755e-06)*u**51
    return f

def a_real_function(u):
    if u < 0.5:
        f = 0.0
    else:
        # f = 1.0
        f = (u - 0.5) * u
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
        y = (0.13246099466340658)*u+(-0.0004964291902550553)*1
    elif u > 0.25 and u < 0.5:
        y = (-0.11134769666265844)*u+(-0.049854125399655946)*1
    elif u >= 0.5 and u < 0.75:
        y = (0.700497543000769)*u+(-0.4101462251187064)*1
    else:
        y = (-0.3920500694198592)*u+(0.008047085874719327)*1
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

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, f_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, f_predict - (f_predict[0] - f_real[0]), label='fix', color='orange', linestyle='--', linewidth=linewith)

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
    file_name = '/' + name + '_r_p_f_function' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_real_and_predict_a_function(name='name'):
    x = []
    N = 100
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    f_real = np.array([a_real_function(ele) for ele in x])
    f_predict = np.array([a_predict_function(ele) for ele in x])

    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, f_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, f_predict, label='fix', color='orange', linestyle='--', linewidth=linewith)

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
               labels=['exact', 'prediction'],  # 'prediction',
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    # ax1.set_facecolor('w')
    plt.show()
    # #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_r_p_a_function' + '.pdf'
    fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)

def plot_real_and_predict_A_function(name='name'):
    A_real = A_real_function()
    A_predict = A_predict_function()
    x = []
    N = 400
    dx = 1.0/N
    for i in range(N + 1):
        x.append(i*dx)
    linewith = 10
    linewith_frame = 4
    title_fontsize = 60
    label_fontsize = 50
    ticks_fontsize = 50

    fig = plt.figure(figsize=(40, 20))
    # plt.plot()
    plot_1, = plt.plot(x, A_real, label='observation', color='red', linestyle='-', linewidth=linewith)
    # plot_2, = plt.plot(x, f_predict, label='prediction', color='orange', linestyle='-', linewidth=linewith)
    plot_3, = plt.plot(x, A_predict, label='fix', color='orange', linestyle='--', linewidth=linewith)

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
               labels=['exact', 'prediction'],  # 'prediction',
               loc="upper left", fontsize=50, frameon=True, edgecolor='green')
    plt.grid(True, which='major', linewidth=0.1, linestyle='--')
    # ax1.set_facecolor('w')
    plt.show()
    #  保存
    save_dir = 'figures/' + name + '_function'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    file_name = '/' + name + '_r_p_A_u_function' + '.pdf'
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
    N = 400
    # test_data_name = 'N_400_example_4_mul_test_another_case_big_1'
    test_data_name = 'beta_300_case_2_0.05'
    real_data_file = 'data/' + test_data_name + '_U' + '.npy'
    # predict_data_file = 'data/' + 'predict_' + test_data_name + '_U' + '.npy'
    predict_data_file = 'data/predict_N_400_example_6_dt_0.1_layer_10_beta_300_U.npy'

    real_time_step = 726
    predict_time_step = 628
    time = 2
    example_index = 0
    real_data = np.load(real_data_file)
    predict_data = np.load(predict_data_file)

    # # 真实和预测以及误差情况
    # plot_real_and_predict_data(real_data, predict_data, example_index, name=test_data_name)

    # 固定时间下真实和预测情况
    # (不含有噪声)
    plot_fixed_time_step(real_data, predict_data, real_time_step, predict_time_step, time, N, example_index, name=test_data_name)
    #（含有噪声）
    # cell_numbers = N
    # real_no_noise_data = np.load('data/N_400_example_2_dt_0.1_layer_10_beta_0.5_no_alpha_U' + '.npy')
    # plot_fixed_time_step_add_noise(real_data, real_no_noise_data, predict_data, real_time_step, predict_time_step, time, example_index, cell_numbers, name=test_data_name)

    # f(u)函数的差别
    # plot_real_and_predict_function(name=test_data_name)
    # plot_real_and_predict_a_function(name=test_data_name)
    # plot_real_and_predict_A_function(name=test_data_name)

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








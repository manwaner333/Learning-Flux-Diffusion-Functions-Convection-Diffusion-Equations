import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

# 加载数据
# data_file = 'data/beta_5_case_3_N_200_0.0_U.npy'
# data = np.load(data_file)[[0, 21, 42, 63, 84, 105, 126, 147, 168, 189], :, :].flatten().reshape(-1, 1)
# data = np.load(data_file)[[0, 21, 42, 63, 84, 105, 126, 147, 168, 189], :, :].reshape((-1, 400))

# data_file = 'data/data_choose_trajectory.npy'
# data = np.load(data_file).reshape((-1, 400))
# data = np.load(data_file).flatten().reshape(-1, 1)

# # 创建TSNE对象，设置参数
# tsne = TSNE(n_components=2, perplexity=5, learning_rate=200)
#
# # 进行t-SNE降维操作
# embedded_data = tsne.fit_transform(data)
#
# # 绘制降维后的数据
# import matplotlib.pyplot as plt
# plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
# plt.show()

# sns.kdeplot(data)
# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.fftpack import fft
#
# # 生成测试数据
# N = 600
# T = 1.0 / 800.0
# t = np.linspace(0.0, N*T, N)
# y = np.sin(50.0 * 2.0*np.pi*t) + 0.5*np.sin(80.0 * 2.0*np.pi*t)
#
# # 进行频谱分析
# yf = fft(y)
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
#
# # 绘制频谱图
# plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
# plt.grid()
# plt.show()

# from scipy import signal
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 生成测试数据
# t = np.linspace(0, 1, 1000, False)
# x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t)
#
# # 设计低通滤波器
# b, a = signal.butter(3, 0.05)
#
# # 应用滤波器
# y = signal.filtfilt(b, a, x)
#
# # 绘制结果
# plt.plot(t, x, 'b', alpha=0.5, label='原始信号')
# plt.plot(t, y, 'r', label='滤波后信号')
# plt.legend()
# plt.grid(True)
# plt.show()


def f_real_function(u):
    f = (np.power(u, 2) * (1 - 5.0 * np.power(1-u, 4))) / (np.power(u, 2) + 0.5 * np.power(1-u, 4))
    return f

obs_points = []
for i in range(0, 399, 20):  # 5
    obs_points.append(i)

# data_file = 'data/beta_5_case_3_N_200_0.0_U.npy'
# data = np.load(data_file)[[0, 21, 42, 63, 84, 105, 126, 147, 168, 189], :, :][:, :, obs_points].flatten()
data_file = 'data/data_choose_trajectory.npy'
data = np.load(data_file)[1:, :, :].flatten()



res = []
for ele in data:
    res.append(f_real_function(ele))

# fig = plt.figure(figsize=(20, 10))
# plt.scatter(data, res, s=5)
# plt.show()
# name = 'scattler'
# save_dir = 'figures/' + name + '_function'
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)
# file_name = '/' + '399_20_standard' + '.pdf'
# fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)


def missing_values(lst):
    result = []
    sub = []
    for i in range(len(lst) - 1):
        sub.append(lst[i])
        if lst[i+1] - lst[i] > 1:
            if len(sub) >= 2:
                result.append(sub)
            sub = []
    if len(sub) > 1:
        result.append(sub)
    return result

def remove_values_above_threshold(arr, threshold):
    return arr[arr <= threshold]

def get_different_stages_der(sub_data, thre=3.5):
    # 0-0.1
    print("the result of 0-0.1")
    where_res_0 = np.where((sub_data >= 0.0) & (sub_data < 0.1))
    places_periods_0 = missing_values(where_res_0[0])
    sub_0 = 0.0
    for ele in places_periods_0:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l)/0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_0 += np.sum(derivatives_c)
    # 0.1-0.2
    print("the result of 0.1-0.2")
    where_res_1 = np.where((sub_data >= 0.1) & (sub_data < 0.2))
    places_periods_1 = missing_values(where_res_1[0])
    sub_1 = 0.0
    for ele in places_periods_1:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_1 += np.sum(derivatives_c)

    # 0.2-0.3
    print("the result of 0.2-0.3")
    where_res_2 = np.where((sub_data >= 0.2) & (sub_data < 0.3))
    places_periods_2 = missing_values(where_res_2[0])
    sub_2 = 0.0
    for ele in places_periods_2:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_2 += np.sum(derivatives_c)

    # 0.3-0.4
    print("the result of 0.3-0.4")
    where_res_3 = np.where((sub_data >= 0.3) & (sub_data < 0.4))
    places_periods_3 = missing_values(where_res_3[0])
    sub_3 = 0.0
    for ele in places_periods_3:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        # print(max(derivatives_c))
        sub_3 += np.sum(derivatives_c)

    # 0.4-0.5
    print("the result of 0.4-0.5")
    where_res_4 = np.where((sub_data >= 0.4) & (sub_data < 0.5))
    places_periods_4 = missing_values(where_res_4[0])
    sub_4 = 0.0
    for ele in places_periods_4:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_4 += np.sum(derivatives_c)

    # 0.5-0.6
    print("the result of 0.5-0.6")
    where_res_5 = np.where((sub_data >= 0.5) & (sub_data < 0.6))
    places_periods_5 = missing_values(where_res_5[0])
    sub_5 = 0.0
    for ele in places_periods_5:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_5 += np.sum(derivatives_c)

    # 0.6-0.7
    print("the result of 0.6-0.7")
    where_res_6 = np.where((sub_data >= 0.6) & (sub_data < 0.7))
    places_periods_6 = missing_values(where_res_6[0])
    sub_6 = 0.0
    for ele in places_periods_6:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_6 += np.sum(derivatives_c)

    # 0.7-0.8
    print("the result of 0.7-0.8")
    where_res_7 = np.where((sub_data >= 0.7) & (sub_data < 0.8))
    places_periods_7 = missing_values(where_res_7[0])
    sub_7 = 0.0
    for ele in places_periods_7:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_7 += np.sum(derivatives_c)

    # 0.8-0.9
    print("the result of 0.8-0.9")
    where_res_8 = np.where((sub_data >= 0.8) & (sub_data < 0.9))
    places_periods_8 = missing_values(where_res_8[0])
    sub_8 = 0.0
    for ele in places_periods_8:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_8 += np.sum(derivatives_c)

    # 0.9-1.0
    print("the result of 0.9-1.0")
    where_res_9 = np.where((sub_data >= 0.9) & (sub_data < 1.0))
    places_periods_9 = missing_values(where_res_9[0])
    sub_9 = 0.0
    for ele in places_periods_9:
        sub_data_l = sub_data[ele[1:]]
        sub_data_r = sub_data[ele[0:-1]]
        derivatives = np.abs((sub_data_r - sub_data_l) / 0.025)
        derivatives_c = remove_values_above_threshold(derivatives, thre)
        print(max(derivatives_c))
        sub_9 += np.sum(derivatives_c)

    return [sub_0, sub_1, sub_2, sub_3, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9]




if __name__ == "__main__":
    # 画出观测数据
    data_file = 'data/beta_5_case_3_N_200_0.0_U.npy'
    data = np.load(data_file)
    obs_time_step = [0, 21, 42, 63, 84, 105, 126, 147, 168, 189]
    N = 400
    x_label = []
    dx = 10/N   # 0.05
    for i in range(400):
        x_label.append(i * dx)
    example_idx = 1
    # plot_1, = plt.plot(x_label, data[obs_time_step[1], example_idx, :], label='observation', color='red',)
    # plot_2, = plt.plot(x_label, data[obs_time_step[2], example_idx, :], label='prediction', color='blue', )
    # plot_3, = plt.plot(x_label, data[obs_time_step[3], example_idx, :], label='observation', color='red',)
    # plot_4, = plt.plot(x_label, data[obs_time_step[4], example_idx, :], label='prediction', color='blue', )
    # plot_5, = plt.plot(x_label, data[obs_time_step[5], example_idx, :], label='observation', color='red',)
    # plot_6, = plt.plot(x_label, data[obs_time_step[6], example_idx, :], label='prediction', color='blue', )
    # plot_7, = plt.plot(x_label, data[obs_time_step[7], example_idx, :], label='observation', color='red',)
    # plot_8, = plt.plot(x_label, data[obs_time_step[8], example_idx, :], label='prediction', color='blue', )
    plot_9, = plt.plot(x_label, data[obs_time_step[9], example_idx, :], label='prediction', color='blue', )
    plt.show()

    # res = (data[obs_time_step[0], example_idx, 1:] - data[obs_time_step[0], example_idx, 0:399])/0.025

    # sub = data[obs_time_step[9], example_idx, :]
    # where_res_0 = np.where((sub >= 0.2) & (sub < 0.3))
    # res = sub[where_res_0]
    # c = (res[1:] - res[0:len(res)-1])/0.025

    sub_data = data[obs_time_step[9], example_idx, :]
    res = get_different_stages_der(sub_data)

    # where_res_0 = np.where((sub_data >= 0.2) & (sub_data < 0.3))
    # places_periods = missing_values(where_res_0[0])
    # for ele in places_periods:
    #     sub_data_l = sub_data[ele[1:]]
    #     sub_data_r = sub_data[ele[0:-1]]
    #     derivatives = np.abs((sub_data_r - sub_data_l)/0.025)
    #     derivatives_c = remove_values_above_threshold(derivatives, 3.5)
    #     sub_sum = np.sum(derivatives_c)
    res_0 = 0
    res_1 = 0
    res_2 = 0
    res_3 = 0
    res_4 = 0
    res_5 = 0
    res_6 = 0
    res_7 = 0
    res_8 = 0
    res_9 = 0

    for i in range(5):
        for j in range(1, 10):
            sub_data = data[obs_time_step[j], i, :]
            res = get_different_stages_der(sub_data, 2.5)
            res_0 += res[0]
            res_1 += res[1]
            res_2 += res[2]
            res_3 += res[3]
            res_4 += res[4]
            res_5 += res[5]
            res_6 += res[6]
            res_7 += res[7]
            res_8 += res[8]
            res_9 += res[9]

    print((res_0, res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8, res_9))

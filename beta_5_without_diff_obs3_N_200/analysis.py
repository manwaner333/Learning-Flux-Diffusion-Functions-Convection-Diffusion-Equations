import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os

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
data = np.load(data_file)[:, :, :].flatten()



res = []
for ele in data:
    res.append(f_real_function(ele))

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 10))
plt.scatter(data, res, s=5)
plt.show()
# name = 'scattler'
# save_dir = 'figures/' + name + '_function'
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)
# file_name = '/' + '399_20_standard' + '.pdf'
# fig.savefig(save_dir + file_name, bbox_inches='tight', pad_inches=0.5)



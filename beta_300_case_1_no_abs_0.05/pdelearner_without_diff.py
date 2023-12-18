from numpy import *
import torch
from torch.autograd import Variable
# from torchviz import make_dot
import expr as expr
from torch.autograd import grad
import numpy as np
import time

__all__ = ['VariantCoeLinear1d']


class VariantCoeLinear1d(torch.nn.Module):
    def __init__(self, T, N, X, batch_size, u0, dt, time_steps, dx, max_f_prime, u_fixed, u_a, N_a, theta, device,
                 is_train=True):
        super(VariantCoeLinear1d, self).__init__()
        coe_num = 1  # the number of coefficient
        self.coe_num = coe_num
        self.T = T
        self.N = N  # The number of grid cell
        self.X = X
        self.batch_size = batch_size
        self.N_a = N_a
        self.theta = theta
        self.allchannels = ['u']
        self.channel_num = 1
        self.hidden_layers = 3
        self.diff_hidden_layers = 2
        polys = []
        for k in range(self.channel_num):
            self.add_module('poly' + str(k), expr.poly(self.hidden_layers, channel_num=len(self.allchannels),
                                                       channel_names=self.allchannels))
            # self.add_module('poly' + str(k), expr.poly(self.hidden_layers, channel_num=len(self.allchannels), channel_names=self.allchannels, theta=self.theta))
            polys.append(self.__getattr__('poly' + str(k)))
        self.polys = tuple(polys)
        self.register_buffer('u0', u0)
        self.register_buffer('u_fixed', u_fixed)
        self.register_buffer('u_a', u_a)
        self.register_buffer('dt', torch.DoubleTensor(1).fill_(dt))
        self.register_buffer('dx', torch.DoubleTensor(1).fill_(dx))
        self.register_buffer('max_f_prime', torch.DoubleTensor(1).fill_(max_f_prime))
        self.register_buffer('time_steps', torch.DoubleTensor(1).fill_(time_steps))
        self.device = device
        self.is_train = is_train

    @property
    def coes(self):
        for i in range(self.coe_num):
            yield self.__getattr__('coe' + str(i))

    @property
    def xy(self):
        return Variable(next(self.coes).inputs)

    @xy.setter
    def xy(self, v):
        for fitter in self.coes:
            fitter.inputs = v

    def coe_params(self):
        parameters = []
        for poly in self.polys:
            parameters += list(poly.parameters())
        for diff in self.diffs:
            parameters += list(diff.parameters())
        return parameters

    def f_predict(self, u):
        u = u.unsqueeze(1)
        Uadd = list(poly(u.permute(0, 2, 1)) for poly in self.polys)
        uadd = torch.cat(Uadd, dim=1)
        return uadd

    def f_real(self, u):
        # f = torch.pow(u, 2) / (torch.pow(u, 2) + 0.5 * torch.pow(1 - u, 2))
        f = 0.5 * u * (3 - torch.pow(u, 2)) + (1 / 12) * 10 * torch.pow(u, 2) * ((3 / 4) - 2 * u + (3 / 2) * torch.pow(u, 2) - (1 / 4) * torch.pow(u, 4))
        return f

    def A_f_half(self, u):
        if self.is_train:
            f = self.f_predict(u)
        else:
            f = self.f_real(u)
        f_left = f[:, 0:self.N - 1]
        f_right = f[:, 1:self.N]
        u_left = u[:, 0:self.N - 1]
        u_right = u[:, 1:self.N]
        # 计算gobal
        b = u.clone().detach()
        b.requires_grad = True
        dfdu = self.df_du(b)
        b.requires_grad = False
        dfdu_left = dfdu[:, 0:self.N - 1]
        dfdu_right = dfdu[:, 1:self.N]
        M = torch.where(dfdu_left > dfdu_right, dfdu_left, dfdu_right)
        f_half = 0.5 * (f_left + f_right) - 0.5 * M * (u_right - u_left)
        # f_half = 0.5 * (f_left + f_right) - 0.5 * self.M * (u_right - u_left)
        return f_half

    def df_du(self, u):
        if self.is_train:
            f = self.f_predict(u)
        else:
            f = self.f_real(u)
        # 计算目前f(u)下面的导数
        dfdu = grad(f, u, grad_outputs=torch.ones_like(f), create_graph=False)[0]
        dfdu = torch.abs(dfdu)
        return dfdu

    def update(self):
        # 计算目前状况下f(u)导数的最大值
        self.u_fixed.requires_grad = True
        dfdu = self.df_du(self.u_fixed)
        max_f_prime = torch.max(dfdu).item()
        self.u_fixed.requires_grad = False
        # if max_f_prime > 0.0 and max_f_prime < 5.0 and amax > 0.0 and amax < 1:
        # max_f_prime = round(max_f_prime, 1)
        dt_a = 0.75 * (self.dx.item() / (max_f_prime + 0.0001))
        n_time = self.T / dt_a
        n_time = int(round(n_time + 1, 0))
        # n_time = max(self.T/dt_a, 20)
        # n_time = int(round(n_time, 0))
        dt = self.T / n_time
        M = max_f_prime
        self.max_f_prime = torch.DoubleTensor(1).fill_(max_f_prime).to(self.device)
        self.M = torch.DoubleTensor(1).fill_(M).to(self.device)
        self.dt = torch.DoubleTensor(1).fill_(dt).to(self.device)
        self.time_steps = torch.IntTensor(1).fill_(n_time).to(self.device)
        print("\033[34mmax_f_prime %.6f, dt %.6f, time_steps %.6f, M%.6f, \033[0m" % (
        self.max_f_prime, self.dt, self.time_steps, M))

    def forward(self, init, stepnum):
        u_old = init
        dt = self.dt
        dx = self.dx
        trajectories = torch.empty((stepnum + 1, self.batch_size, self.N), requires_grad=False, device=self.device)
        trajectories[0, :, :] = u_old
        for i in range(1, stepnum + 1):
            t1 = time.time()
            f_half = self.A_f_half(u_old)
            t2 = time.time()
            u = torch.empty((self.batch_size, self.N), requires_grad=False).to(self.device)
            f_half_left = f_half[:, 0:self.N - 2]
            f_half_right = f_half[:, 1:self.N - 1]
            u_old_sub = u_old[:, 1:self.N - 1]
            u[:, 1:self.N - 1] = u_old_sub - (dt / dx) * (f_half_right - f_half_left)
            u[:, 0] = u[:, 1]
            u[:, self.N - 1] = u[:, self.N - 2]
            u_old = u
            # print((u_old.min(), u_old.max()))
            trajectories[i, :] = u_old
            t3 = time.time()
            # print(f'{"time_interval_1"}: {t2 - t1:.4f}s')
            # print(f'{"time_interval_2"}: {t3 - t1:.4f}s')
        return trajectories


def generate_real_data(save_file, u0_file, u_fixed_file, ua_file):
    device = 'cpu'
    T = 2.0
    X = 10
    N = 400  # 200
    dx = X / N
    dt = 0.08
    time_steps = 200
    max_f_prime = -0.03
    theta = 0.0001
    # u_0
    batch_size = 4
    u_0_np = np.zeros((batch_size, N), dtype=float)
    u_0_np[0:1, 160:240] = 1.0
    u_0_np[1:2, 160:240] = 0.8
    u_0_np[2:3, 140:200] = 0.7
    u_0_np[3:4, 180:260] = 0.9
    # u_0_np[1:2, 80:120] = 1.0
    # u_0_np[2:3, 70:110] = 0.9
    # u_0_np[4:5, 70:110] = 0.6
    u_0 = torch.from_numpy(u_0_np)
    u_0 = u_0.to(device)
    # 引入 u_fixed, 用来计算max_f_prime
    # du = 1.0/100
    # u_fixed_0 = 0.0
    # u_fixed_np = np.zeros((1, 101), dtype=float)
    # u_fixed_np[:1, 0] = u_fixed_0
    # for i in range(1, 101):
    #     u_fixed_np[:1, i] = u_fixed_0 + i * du
    # u_fixed = torch.from_numpy(u_fixed_np)
    # u_fixed = u_fixed.to(device)
    du = 1.2 / 52
    u_fixed_0 = -0.1 + 0.5 * du
    u_fixed_np = np.zeros((1, 52), dtype=float)
    u_fixed_np[:1, 0] = u_fixed_0
    for i in range(1, 52):
        u_fixed_np[:1, i] = u_fixed_0 + i * du
    u_fixed = torch.from_numpy(u_fixed_np)
    u_fixed = u_fixed.to(device)
    # 计算a的积分的
    N_a = 400
    u_a_np = np.zeros((batch_size, N_a + 1), dtype=float)
    u_a_0 = 0.0
    u_a_np[:, 0] = u_a_0
    for i in range(1, N_a + 1):
        u_a_np[:, i] = u_a_0 + (1.0 / N_a) * i
    u_a = torch.from_numpy(u_a_np)
    u_a = u_a.to(device)

    # model
    linpdelearner = VariantCoeLinear1d(T=T, N=N, X=X, batch_size=batch_size, u0=u_0, dt=dt, time_steps=time_steps
                                       , dx=dx, max_f_prime=max_f_prime, u_fixed=u_fixed, u_a=u_a, N_a=N_a,
                                       theta=theta, device=device, is_train=False)
    # 预测值
    linpdelearner.update()
    U = linpdelearner(linpdelearner.u0, linpdelearner.time_steps)
    print("U")
    print(U.shape)
    print("u_0")
    print(u_0.shape)
    print("u_fixed")
    print(u_fixed.shape)
    np.save(save_file, U.detach().to('cpu'))
    np.save(u0_file, u_0.detach().to('cpu'))
    np.save(u_fixed_file, u_fixed.detach().to('cpu'))
    np.save(ua_file, u_a.detach().to('cpu'))


if __name__ == "__main__":
    experiment_name = 'N_400_example_4_mul_test_double_without_diff'
    real_data_file = 'data/' + experiment_name + '_U' + '.npy'
    u0_file = 'data/' + experiment_name + '_u0' + '.npy'
    u_fixed_file = 'data/' + experiment_name + '_u_fixed' + '.npy'
    ua_file = 'data/' + experiment_name + '_ua' + '.npy'
    generate_real_data(real_data_file, u0_file, u_fixed_file, ua_file)

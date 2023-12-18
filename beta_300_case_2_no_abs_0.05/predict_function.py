from numpy import *
import torch
from torch.autograd import Variable
# from torchviz import make_dot
import expr
# import expr_diffusion
import expr_diffusion_1
import expr_diffusion_2
import expr_diffusion_3
import expr_diffusion_4
from torch.autograd import grad
import numpy as np
from interp1d import Interp1d

__all__ = ['VariantCoeLinear1d']


class VariantCoeLinear1d(torch.nn.Module):
    def __init__(self, T, N, X, batch_size, u0, dt, time_steps, dx, max_f_prime, u_fixed, u_a, N_a, theta, device,
                 is_train=True):
        super(VariantCoeLinear1d, self).__init__()
        coe_num = 1
        self.coe_num = coe_num
        self.T = T
        self.N = N
        self.X = X
        self.batch_size = batch_size
        self.N_a = N_a
        self.theta = theta
        self.allchannels = ['u']
        self.channel_num = 1
        self.hidden_layers = 6
        self.diff_hidden_layers_1 = 0
        self.diff_hidden_layers_2 = 0
        self.diff_hidden_layers_3 = 0
        self.diff_hidden_layers_4 = 0
        polys = []
        for k in range(self.channel_num):
            self.add_module('poly' + str(k), expr.poly(self.hidden_layers, channel_num=len(self.allchannels),
                                                       channel_names=self.allchannels))
            polys.append(self.__getattr__('poly' + str(k)))
        self.polys = tuple(polys)
        diffs1 = []
        for k in range(self.channel_num):
            self.add_module('diff1' + str(k),
                            expr_diffusion_1.poly(self.diff_hidden_layers_1, channel_num=len(self.allchannels),
                                                  channel_names=self.allchannels))
            diffs1.append(self.__getattr__('diff1' + str(k)))
        diffs2 = []
        for k in range(self.channel_num):
            self.add_module('diff2' + str(k),
                            expr_diffusion_2.poly(self.diff_hidden_layers_2, channel_num=len(self.allchannels),
                                                  channel_names=self.allchannels))
            diffs2.append(self.__getattr__('diff2' + str(k)))

        diffs3 = []
        for k in range(self.channel_num):
            self.add_module('diff3' + str(k),
                            expr_diffusion_3.poly(self.diff_hidden_layers_3, channel_num=len(self.allchannels),
                                                  channel_names=self.allchannels))
            diffs3.append(self.__getattr__('diff3' + str(k)))
        diffs4 = []
        for k in range(self.channel_num):
            self.add_module('diff4' + str(k),
                            expr_diffusion_4.poly(self.diff_hidden_layers_4, channel_num=len(self.allchannels),
                                                  channel_names=self.allchannels))
            diffs4.append(self.__getattr__('diff4' + str(k)))
        self.diffs1 = tuple(diffs1)
        self.diffs2 = tuple(diffs2)
        self.diffs3 = tuple(diffs3)
        self.diffs4 = tuple(diffs4)
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
        for diff in self.diffs1:
            parameters += list(diff.parameters())
        for diff in self.diffs2:
            parameters += list(diff.parameters())
        for diff in self.diffs3:
            parameters += list(diff.parameters())
        for diff in self.diffs4:
            parameters += list(diff.parameters())
        return parameters

    def f_predict(self, u):
        u = u.unsqueeze(1)
        Uadd = list(poly(u.permute(0, 2, 1)) for poly in self.polys)
        uadd = torch.cat(Uadd, dim=1)
        return uadd

    def f_real(self, u):
        f = (-25972.371598619586)*u**25+(25840.812343138885)*u**24+(-19884.140782744547)*u**20+(17085.345069783354)*u**26+(16357.674571102249)*u**21+(-14825.732069733394)*u**23+(13501.289357667172)*u**19+(8645.249702576904)*u**15+(-7236.374039981482)*u**14+(7102.099097157273)*u**29+(-5493.341258068011)*u**16+(-5219.3700543720215)*u**30+(-5162.229419639262)*u**27+(-5121.4548944559865)*u**18+(5007.950820777607)*u**34+(-4897.029399809083)*u**33+(-3982.450361365026)*u**35+(-3865.0478416226433)*u**28+(2894.029956532971)*u**32+(2665.6699177649016)*u**13+(2649.0277282712677)*u**36+(2356.5193091271185)*u**17+(-2334.901552054318)*u**22+(-1522.570290185906)*u**37+(965.964494891417)*u**31+(-920.5920884029395)*u**11+(769.6499860336208)*u**38+(670.4323238231805)*u**12+(650.4757696247805)*u**9+(-345.8635858606117)*u**39+(250.29415112615064)*u**6+(-231.45564683774361)*u**7+(-217.14302316246128)*u**8+(-204.15745327222774)*u**10+(139.140488682436)*u**40+(-90.73948481593175)*u**5+(-50.34902781777148)*u**41+(-42.633920146288844)*u**3+(38.2010108054874)*u**4+(16.868944239777186)*u**2+(16.44035687554242)*u**42+(-4.8543850866942835)*u**43+(1.6251119401045502)*u+(1.2978165520437026)*u**44+(-0.3699464487222815)*1+(-0.31433603585388764)*u**45+(0.06896670683361808)*u**46+(-0.013696797405731096)*u**47+(0.0024584198521216682)*u**48+(-0.00039775227157072264)*u**49+(5.7768516450493116e-05)*u**50+(-7.4838375300789755e-06)*u**51
        return f

    # # 调整成为分段函数
    def u1(self, u):
        u1 = torch.where(u >= -0.1, 1.0, 0.0)
        u2 = torch.where(u < 0.25, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u2(self, u):
        u1 = torch.where(u >= 0.25, 1.0, 0.0)
        u2 = torch.where(u < 0.5, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u3(self, u):
        u1 = torch.where(u >= 0.5, 1.0, 0.0)
        u2 = torch.where(u < 0.75, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def u4(self, u):
        u1 = torch.where(u >= 0.75, 1.0, 0.0)
        u2 = torch.where(u <= 1.1, 1.0, 0.0)
        res = u1.mul(u2)
        return res

    def a_real_1(self, u):
        a = (0.13246099466340658)*u+(-0.0004964291902550553)*1
        return a.double()

    def a_real_2(self, u):
        a = (-0.11134769666265844)*u+(-0.049854125399655946)*1
        return a.double()

    def a_real_3(self, u):
        a = (0.700497543000769)*u+(-0.4101462251187064)*1
        return a.double()

    def a_real_4(self, u):
        a = (-0.3920500694198592)*u+(0.008047085874719327)*1
        return a.double()

    def a_real(self, u):
        a = self.a_real_1(u) * self.u1(u) + self.a_real_2(u) * self.u2(u) + self.a_real_3(u) * self.u3(
            u) + self.a_real_4(u) * self.u4(u)
        return torch.abs(a).double()

    def a_predict(self, u):
        u = u.unsqueeze(1)
        u1 = self.u1(u)
        u2 = self.u2(u)
        u3 = self.u3(u)
        u4 = self.u4(u)
        Uadd1 = list(diff(u.permute(0, 2, 1)) for diff in self.diffs1)
        uadd1 = torch.cat(Uadd1, dim=1)
        Uadd2 = list(diff(u.permute(0, 2, 1)) for diff in self.diffs2)
        uadd2 = torch.cat(Uadd2, dim=1)
        Uadd3 = list(diff(u.permute(0, 2, 1)) for diff in self.diffs3)
        uadd3 = torch.cat(Uadd3, dim=1)
        Uadd4 = list(diff(u.permute(0, 2, 1)) for diff in self.diffs4)
        uadd4 = torch.cat(Uadd4, dim=1)
        uadd = uadd1 * u1[0] + uadd2 * u2[0] + uadd3 * u3[0] + uadd4 * u4[0]
        # 增加一句
        # uadd = torch.where(uadd >=  0.0, uadd, 0.0)
        return torch.abs(uadd)

    def a_max(self, u):
        if self.is_train:
            a = 0.05 * self.a_predict(u)
        else:
            a = 0.05 * self.a_real(u)
        return torch.max(a)

    def A_f_half(self, u):
        if self.is_train:
            f = self.f_predict(u)
            a = 0.05 * self.a_predict(self.u_a)
        else:
            f = self.f_real(u)
            a = 0.05 * self.a_real(self.u_a)
        f_left = f[:, 0:self.N - 1]
        f_right = f[:, 1:self.N]
        u_left = u[:, 0:self.N - 1]
        u_right = u[:, 1:self.N]
        # 计算gobal
        b = u.clone().detach()
        b.requires_grad = True
        dfdu = self.df_du(b)
        b.requires_grad = False
        # 修改前
        dfdu_left = dfdu[:, 0:self.N - 1]
        dfdu_right = dfdu[:, 1:self.N]
        M = torch.where(dfdu_left > dfdu_right, dfdu_left, dfdu_right)
        # 修改后
        # M = torch.max(dfdu)
        f_half = 0.5 * (f_left + f_right) - 0.5 * M * (u_right - u_left)
        A_a = torch.empty((self.batch_size, self.N_a + 1), requires_grad=False).to(self.device)
        for index in range(1, self.N_a + 2):
            A_a[:, index - 1] = torch.trapz(a[:, 0:index], self.u_a[:, 0:index])
        A_u = None
        A_u = Interp1d()(self.u_a, A_a, u, A_u)
        A_u_left = A_u[:, 0:self.N - 1]
        A_u_right = A_u[:, 1:self.N]
        A_half = (A_u_right - A_u_left) / self.dx
        return f_half, A_half

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
        # 计算a(u)的最大值
        amax = self.a_max(self.u_fixed).item()
        print("\033[32mmax_f_prime %.6f, amax%.6f, \033[0m" % (max_f_prime, amax))
        if 0.0 < max_f_prime < 25.0 and 0.0 < amax < 0.5:
            dt_a = 0.75 * (self.dx.item() / (max_f_prime + 0.0001 + 2 * amax / self.dx.item()))
            n_time = self.T / dt_a
            n_time = int(round(n_time + 1, 0))
            dt = self.T / n_time
            M = max_f_prime
            self.max_f_prime = torch.DoubleTensor(1).fill_(max_f_prime).to(self.device)
            self.M = torch.DoubleTensor(1).fill_(M).to(self.device)
            self.dt = torch.DoubleTensor(1).fill_(dt).to(self.device)
            self.time_steps = torch.IntTensor(1).fill_(n_time).to(self.device)
            print("\033[34mmax_f_prime %.6f, dt %.6f, time_steps %.6f, amax%.6f, M%.6f, \033[0m" % (self.max_f_prime,
                                                                                                    self.dt, self.time_steps, amax, M))

    def forward(self, init, stepnum):
        u_old = init
        dt = self.dt
        dx = self.dx
        trajectories = torch.empty((stepnum + 1, self.batch_size, self.N), requires_grad=False, device=self.device)
        trajectories[0, :, :] = u_old
        for i in range(1, stepnum + 1):
            f_half, A_half = self.A_f_half(u_old)
            u = torch.empty((self.batch_size, self.N), requires_grad=False).to(self.device)
            f_half_left = f_half[:, 0:self.N - 2]
            f_half_right = f_half[:, 1:self.N - 1]
            A_half_left = A_half[:, 0:self.N - 2]
            A_half_right = A_half[:, 1:self.N - 1]
            u_old_sub = u_old[:, 1:self.N - 1]
            u[:, 1:self.N - 1] = u_old_sub - (dt / dx) * (f_half_right - f_half_left) + (dt / dx) * (A_half_right - A_half_left)
            u[:, 0] = u[:, 1]
            u[:, self.N - 1] = u[:, self.N - 2]
            u_old = u
            trajectories[i, :] = u_old
        return trajectories


def generate_real_data(save_file):
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
    # batch_size = 4
    # u_0_np = np.zeros((batch_size, N), dtype=float)
    # u_0_np[0:1, 160:240] = 1.0
    # u_0_np[1:2, 160:240] = 0.8
    # u_0_np[2:3, 140:200] = 0.7
    # u_0_np[3:4, 180:260] = 0.9
    # u_0 = torch.from_numpy(u_0_np)
    # u_0 = u_0.to(device)
    batch_size = 6
    u_0_np = np.zeros((batch_size, N), dtype=float)
    u_0_np[0:1, 160:240] = 1.0
    u_0_np[1:2, 160:240] = 0.9
    u_0_np[2:3, 140:200] = 0.8
    u_0_np[3:4, 180:260] = 0.7
    u_0_np[4:5, 0:80] = 0.85
    u_0_np[5:6, 0:80] = 0.75
    u_0 = torch.from_numpy(u_0_np)
    u_0 = u_0.to(device)
    # 引入 u_fixed, 用来计算max_f_prime
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


if __name__ == "__main__":
    # experiment_name = 'example_6_hidden_layer_6_no_abs_2'
    experiment_name = 'beta_300_case_2_0.05'
    real_data_file = 'data/' + 'predict_' + experiment_name + '_U' + '.npy'
    generate_real_data(real_data_file)

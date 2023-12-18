#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, contextlib
import numpy as np
import torch
import getopt, yaml, time
import pdelearner

np.random.seed(0)
torch.manual_seed(0)
from torch import nn
from interp1d import Interp1d

mse = nn.MSELoss()


def _options_cast(options, typeset, thistype):
    for x in typeset:
        options['--' + x] = thistype(options['--' + x])
    return options


def _option_analytic(option, thistype):
    if not isinstance(option, str):
        return option
    l0 = option.split(',')
    l = []
    for l1 in l0:
        try:
            ll = thistype(l1)
            x = [ll, ]
        except ValueError:
            z = l1.split('-')
            x = list(range(int(z[0]), int(z[1]) + 1))
        finally:
            l = l + x
    return l


def _setoptions(options):
    assert options['--precision'] in ['float', 'double']
    # str options
    strtype = ['taskdescriptor', 'recordfile', 'device']
    options = _options_cast(options, strtype, str)
    inttype = ['batch_size', 'maxiter', 'recordcycle', 'savecycle', 'time_steps', 'layer']
    options = _options_cast(options, inttype, int)
    # float options
    floattype = ['dt', 'T', 'X']
    options = _options_cast(options, floattype, float)
    return options


def setoptions(*, argv=None, kw=None, configfile=None, isload=False):
    """
    proirity: argv>kw>configfile
    Arguments:
        argv (list): command line options
        kw (dict): options
        configfile (str): configfile path
        isload (bool): load or set new options
    """
    options = {
        '--precision': 'double',
        '--xn': '50',
        '--yn': '50',
    }
    longopts = list(k[2:] + '=' for k in options)
    longopts.append('configfile=')
    if not argv is None:
        options.update(dict(getopt.getopt(argv, shortopts='f', longopts=longopts)[0]))
    if '--configfile' in options:
        assert configfile is None, 'duplicate configfile in argv.'
        configfile = options['--configfile']
    if not configfile is None:
        options['--configfile'] = configfile
        with open(configfile, 'r') as f:
            options.update(yaml.safe_load(f))
    if not kw is None:
        options.update(kw)
    if not argv is None:
        options.update(dict(getopt.getopt(argv, shortopts='f', longopts=longopts)[0]))
    options = _setoptions(options)
    options.pop('-f', 1)
    savepath = 'checkpoint/' + options['--taskdescriptor']
    if not isload:
        try:
            os.makedirs(savepath)
        except FileExistsError:
            os.rename(savepath, savepath + '-' + str(np.random.randint(2 ** 32)))
            os.makedirs(savepath)
        with open(savepath + '/options.yaml', 'w') as f:
            print(yaml.dump(options), file=f)
    return options


class callbackgen(object):
    def __init__(self, options, nfi=None, module=None, stage=None):
        self.taskdescriptor = options['--taskdescriptor']
        self.recordfile = options['--recordfile']
        self.recordcycle = options['--recordcycle']
        self.savecycle = options['--savecycle']
        self.savepath = 'checkpoint/' + self.taskdescriptor
        self.startt = time.time()
        self.Fs = []
        self.Gs = []
        self.ITERNUM = 0

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, v):
        self._stage = v
        with self.open() as output:
            print('\n', file=output)
            print('current stage is: ' + v, file=output)

    @contextlib.contextmanager
    def open(self):
        isfile = (not self.recordfile is None)
        if isfile:
            output = open(self.savepath + '/' + self.recordfile, 'a')
        else:
            output = sys.stdout
        try:
            yield output
        finally:
            if isfile:
                output.close()

    # remember to set self.nfi,self.module,self.stage
    def save(self, xopt, iternum):
        self.nfi.flat_params = xopt
        try:
            os.mkdir(self.savepath + '/params')
        except:
            pass
        filename = self.savepath + '/params/' + str(self.stage) + '-xopt-' + str(iternum)
        torch.save(self.module.state_dict(), filename)
        return None

    def load(self, l, iternum=None):
        """
        load storaged parameters from a file.
        the name of the file from which we will load
        is determined by l and iternum
        """
        if l == 0:
            stage = 'warmup'
        else:
            stage = 'layer-' + str(l)
        if iternum is None:
            iternum = 'final'
        else:
            iternum = str(iternum)
        filename = self.savepath + '/params/' + str(stage) + '-xopt-' + iternum
        params = torch.load(filename)
        self.module.load_state_dict(params)
        return None

    def record(self, xopt, iternum, **args):
        self.Fs.append(self.nfi.f(xopt))
        self.Gs.append(np.linalg.norm(self.nfi.fprime(xopt)))
        stopt = time.time()
        with self.open() as output:
            print('iter:{:6d}'.format(iternum), '   time: {:.2f}'.format(stopt - self.startt), file=output)
            print('Func: {:.2e}'.format(self.Fs[-1]), ' |g|: {:.2e}'.format(self.Gs[-1]), file=output)
        self.startt = stopt
        return None

    def __call__(self, xopt, **args):
        if self.ITERNUM % self.recordcycle == 0:
            self.record(xopt, iternum=self.ITERNUM, **args)
        if self.ITERNUM % self.savecycle == 0:
            self.save(xopt, iternum=self.ITERNUM)
        self.ITERNUM += 1
        return None


def setenv(options):
    namestobeupdate = {}
    namestobeupdate['device'] = options['--device']
    namestobeupdate['precision'] = options['--precision']
    namestobeupdate['taskdescriptor'] = options['--taskdescriptor']
    namestobeupdate['batch_size'] = options['--batch_size']
    namestobeupdate['maxiter'] = options['--maxiter']
    namestobeupdate['T'] = options['--T']
    namestobeupdate['X'] = options['--X']
    namestobeupdate['dt'] = options['--dt']
    namestobeupdate['dx'] = options['--dx']
    namestobeupdate['time_steps'] = options['--time_steps']
    namestobeupdate['N'] = options['--N']
    namestobeupdate['N_a'] = options['--N_a']
    namestobeupdate['layer'] = options['--layer']
    namestobeupdate['recordfile'] = options['--recordfile']
    namestobeupdate['recordcycle'] = options['--recordcycle']
    namestobeupdate['savecycle'] = options['--savecycle']
    namestobeupdate['theta'] = options['--theta']

    # 引入u_0
    u_0_file = 'data/' + namestobeupdate['taskdescriptor'] + '_u0' + '.npy'
    u_0 = torch.from_numpy(np.load(u_0_file))
    u_0 = u_0.to(namestobeupdate['device'])

    # 引入 u_fixed, 用来计算max_f_prime
    u_fixed_file = 'data/' + namestobeupdate['taskdescriptor'] + '_u_fixed' + '.npy'
    u_fixed = torch.from_numpy(np.load(u_fixed_file))
    u_fixed = u_fixed.to(namestobeupdate['device'])

    # 引入u_a
    u_a_file = 'data/' + namestobeupdate['taskdescriptor'] + '_ua' + '.npy'
    u_a = torch.from_numpy(np.load(u_a_file))
    u_a = u_a.to(namestobeupdate['device'])

    # 引入max_f_prime
    max_f_prime = 0.1
    linpdelearner = pdelearner.VariantCoeLinear1d(T=namestobeupdate['T'], N=namestobeupdate['N'],
                                                  X=namestobeupdate['X'],
                                                  batch_size=namestobeupdate['batch_size'], u0=u_0,
                                                  dt=namestobeupdate['dt'], time_steps=namestobeupdate['time_steps'],
                                                  dx=namestobeupdate['dx'], max_f_prime=max_f_prime, u_fixed=u_fixed,
                                                  u_a=u_a, N_a=namestobeupdate['N_a'], theta=namestobeupdate['theta'],
                                                  device=namestobeupdate['device'], is_train=True)

    if namestobeupdate['precision'] == 'double':
        linpdelearner.double()
    else:
        linpdelearner.float()

    linpdelearner.to(namestobeupdate['device'])
    callback = callbackgen(options)
    callback.module = linpdelearner

    return namestobeupdate, callback, linpdelearner


def _sparse_loss(model):
    """
    SymNet regularization
    """
    loss = 0
    s = 1e-2
    for p in model.coe_params():
        p = p.abs()
        loss = loss + ((p < s).to(p) * 0.5 / s * p ** 2).sum() + ((p >= s).to(p) * (p - s / 2)).sum()
    return loss


def print_model_parameters(linpdelearner):
    for parameters in linpdelearner.parameters():
        print(parameters)


def printcoeffs(linpdelearner):
    for poly in linpdelearner.polys:
        tsym, csym = poly.coeffs()
        str_molecular = '(' + str(csym[0]) + ')' + '*' + str(tsym[0])
        for index in range(1, len(tsym)):
            str_molecular += '+' + '(' + str(csym[index]) + ')' + '*' + str(tsym[index])
        print(str_molecular)
    for diff in linpdelearner.diffs1:
        tsym, csym = diff.coeffs()
        str_molecular = '(' + str(csym[0]) + ')' + '*' + str(tsym[0])
        for index in range(1, len(tsym)):
            str_molecular += '+' + '(' + str(csym[index]) + ')' + '*' + str(tsym[index])
        print(str_molecular)
    for diff in linpdelearner.diffs2:
        tsym, csym = diff.coeffs()
        str_molecular = '(' + str(csym[0]) + ')' + '*' + str(tsym[0])
        for index in range(1, len(tsym)):
            str_molecular += '+' + '(' + str(csym[index]) + ')' + '*' + str(tsym[index])
        print(str_molecular)
    for diff in linpdelearner.diffs3:
        tsym, csym = diff.coeffs()
        str_molecular = '(' + str(csym[0]) + ')' + '*' + str(tsym[0])
        for index in range(1, len(tsym)):
            str_molecular += '+' + '(' + str(csym[index]) + ')' + '*' + str(tsym[index])
        print(str_molecular)
    for diff in linpdelearner.diffs4:
        tsym, csym = diff.coeffs()
        str_molecular = '(' + str(csym[0]) + ')' + '*' + str(tsym[0])
        for index in range(1, len(tsym)):
            str_molecular += '+' + '(' + str(csym[index]) + ')' + '*' + str(tsym[index])
        print(str_molecular)


def specific_time_solution(U, xt, t, dt, x0):
    time_step = round(t/dt) + 1
    U_sub = U[time_step, :, :]
    U_adj = None
    U_adj = Interp1d()(x0, U_sub, xt, U_adj)
    return U_adj



def loss(model, stepnum, obs_data, layerweight=None):
    # 注意这个地方 stepnum 和 model.time_steps 是不一样的
    if layerweight is None:
        layerweight = [1, ] * stepnum
        layerweight[-1] = 1
    ut = model.u0
    stableloss = 0
    sparseloss = _sparse_loss(model)

    # 模型更新
    model.update()

    # 调试：打印优化过程中的模型
    print_model_parameters(model)
    printcoeffs(model)

    # 时间list
    obs_t = []
    for i in range(stepnum):
        obs_t.append(0.1 * i)

    # 真实数据， 步骤list
    dt_fixed = 0.002198
    obs_time_step = []

    for ele in obs_t:
        obs_time_step.append(round(ele / dt_fixed))
    # obs_data_choose = obs_data[obs_time_step, :, :]

    # 预测数据， 步骤list
    dt_changed = model.dt.item()
    pre_time_step = []
    for ele in obs_t:
        pre_time_step.append(round(ele / dt_changed))

    # 预测的轨迹
    trajectories = model(ut, pre_time_step[-1] + 1)
    # pre_data_choose = trajectories[pre_time_step, :, :]

    x0_file = 'data/beta_300_case_2_N_200_0.5_obs_3_x0.npy'
    x0_data = torch.from_numpy(np.load(x0_file))
    x0_data = x0_data.to(model.device)

    # 选取真实数据集
    obs_data_choose = torch.empty((len(obs_t), model.batch_size, model.N), requires_grad=False, device=model.device)
    xi_old = x0_data
    for i in range(len(obs_t)):
        if obs_t[i] == 0:
            obs_data_choose[i, :, :] = xi_old
        else:
            ads_U = specific_time_solution(obs_data, xi_old, obs_t[i], dt_fixed, x0_data)
            xi = 1 * ads_U * 0.1 + xi_old
            xi_old = xi
            obs_data_choose[i, :, :] = xi_old

    # 选取预测数据集
    pre_data_choose = torch.empty((len(obs_t), model.batch_size, model.N), requires_grad=False, device=model.device)
    xi_old = x0_data
    for i in range(len(obs_t)):
        if obs_t[i] == 0:
            pre_data_choose[i, :, :] = xi_old
        else:
            model_dt = model.dt.item()
            ads_U_pre = specific_time_solution(trajectories, xi_old, obs_t[i], model_dt, x0_data)
            xi = 1 * ads_U_pre * 0.1 + xi_old
            xi_old = xi
            pre_data_choose[i, :, :] = xi_old

    # 常微分的求解方法:
    # (1) x_1 = x_0 + Delta t u(x_0, t_0)
    # (2) x_2 = x_1 + Delta t u(x_1, t_1)
    # (3) x_3 = x_2 + Delta t u(x_2, t_2)


    # 自己写的关于f(u)的导数
    du = 1.0 / 500
    u_fixed_0_1 = -0.002
    u_fixed_np_1 = np.zeros((1, 502), dtype=float)
    u_fixed_np_1[:1, 0] = u_fixed_0_1
    for i in range(1, 502):
        u_fixed_np_1[:1, i] = u_fixed_0_1 + i * du
    u_fixed_1 = torch.from_numpy(u_fixed_np_1)
    u_fixed_1 = u_fixed_1.to(model.device)
    f_test_1 = model.f_predict(u_fixed_1)
    dfdu_1 = torch.empty((1, 501), requires_grad=False).to(model.device)
    for i in range(0, 501):
        dfdu_1[:1, i] = ((f_test_1[:, i + 1] - f_test_1[:, i]) / du)
    dfdu_1 = dfdu_1.to(dtype=torch.float64)
    max_f_prime = torch.max(torch.abs(dfdu_1))

    # 求出预测的a(u)
    du_a = 1.0 / 400
    u_fixed_a_0 = 0.0
    u_fixed_a_np = np.zeros((1, 401), dtype=float)
    u_fixed_a_np[:1, 0] = u_fixed_a_0
    for i in range(1, 401):
        u_fixed_a_np[:1, i] = u_fixed_a_0 + i * du_a
    u_fixed_a = torch.from_numpy(u_fixed_a_np)
    u_fixed_a = u_fixed_a.to(model.device)
    a = 0.5 * model.a_predict(u_fixed_a)
    # amin
    amin = torch.min(a)
    # amax
    amax = torch.max(a)
    # aloss
    aloss = torch.abs(torch.where(a <= 0.0, a, 0.0)).sum()

    # 位置观测点
    obs_points = []
    for i in range(0, 199, 15):  # 5
        obs_points.append(i)

    amax_up_bound = 10.0
    fmax_up_bound = 25.0
    if 0 < max_f_prime < fmax_up_bound and 0 < amax < amax_up_bound:
        print("mormal")
        loss = mse(obs_data_choose[:, :, obs_points], pre_data_choose[:, :, obs_points])
    elif 0 < max_f_prime < fmax_up_bound and (amax <= 0.0 or amax >= amax_up_bound):
        print("a#loss")
        loss = torch.abs(amax - amax_up_bound) + torch.abs(amax)
    elif (max_f_prime <= 0 or max_f_prime >= fmax_up_bound) and 0.0 < amax < amax_up_bound:
        print("f#loss")
        loss = torch.abs(max_f_prime - fmax_up_bound) + torch.abs(max_f_prime)
    else:
        print("f#a#loss")
        loss = torch.abs(max_f_prime - fmax_up_bound) + torch.abs(max_f_prime) + torch.abs(amax - amax_up_bound) + torch.abs(amax)

    # 打印相关的数据
    print('obs_time_step:')
    print(obs_time_step)
    print("dt")
    print(dt_changed)
    print('pre_time_step:')
    print(pre_time_step)

    print("\033[33m loss0 %.6f, max_f_prime %.6f, amin %.6f, amax %.6f, a_loss %.6f \033[0m" % (
            loss, max_f_prime, amin, amax, aloss))

    return loss, sparseloss, stableloss

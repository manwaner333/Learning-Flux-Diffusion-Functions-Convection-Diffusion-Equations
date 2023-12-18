#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, contextlib
import numpy as np
import torch
import getopt, yaml, time
import pdelearner
from torch.autograd import grad
np.random.seed(0)
torch.manual_seed(0)
from torch import nn
mse = nn.MSELoss()

def _options_cast(options, typeset, thistype):
    for x in typeset:
        options['--'+x] = thistype(options['--'+x])
    return options
def _option_analytic(option, thistype):
    if not isinstance(option, str):
        return option
    l0 = option.split(',')
    l = []
    for l1 in l0:
        try:
            ll = thistype(l1)
            x = [ll,]
        except ValueError:
            z = l1.split('-')
            x = list(range(int(z[0]), int(z[1])+1))
        finally:
            l = l+x
    return l
def _setoptions(options):
    assert options['--precision'] in ['float','double']
    # str options
    strtype = ['taskdescriptor', 'recordfile', 'device']
    options = _options_cast(options, strtype, str)
    inttype = ['batch_size', 'maxiter', 'recordcycle', 'savecycle', 'time_steps', 'layer']
    options = _options_cast(options, inttype, int)
    # float options
    floattype = ['dt', 'T', 'X']
    options = _options_cast(options, floattype, float)
    # options['--layer'] = list(_option_analytic(options['--layer'], int))
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
    longopts = list(k[2:]+'=' for k in options)
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
    savepath = 'checkpoint/'+options['--taskdescriptor']
    if not isload:
        try:
            os.makedirs(savepath)
        except FileExistsError:
            os.rename(savepath, savepath+'-'+str(np.random.randint(2**32)))
            os.makedirs(savepath)
        with open(savepath+'/options.yaml', 'w') as f:
            print(yaml.dump(options), file=f)
    return options

class callbackgen(object):
    def __init__(self, options, nfi=None, module=None, stage=None):
        self.taskdescriptor = options['--taskdescriptor']
        self.recordfile = options['--recordfile']
        self.recordcycle = options['--recordcycle']
        self.savecycle = options['--savecycle']
        self.savepath = 'checkpoint/'+self.taskdescriptor
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
            print('current stage is: '+v, file=output)
    @contextlib.contextmanager
    def open(self):
        isfile = (not self.recordfile is None)
        if isfile:
            output = open(self.savepath+'/'+self.recordfile, 'a')
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
            os.mkdir(self.savepath+'/params')
        except:
            pass
        filename = self.savepath+'/params/'+str(self.stage)+'-xopt-'+str(iternum)
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
            stage = 'layer-'+str(l)
        if iternum is None:
            iternum = 'final'
        else:
            iternum = str(iternum)
        filename = self.savepath+'/params/'+str(stage)+'-xopt-'+iternum
        # params = torch.load(filename, map_location=self.module.device)
        params = torch.load(filename)
        self.module.load_state_dict(params)
        return None

    def record(self, xopt, iternum, **args):
        self.Fs.append(self.nfi.f(xopt))
        self.Gs.append(np.linalg.norm(self.nfi.fprime(xopt)))
        stopt = time.time()
        with self.open() as output:
            print('iter:{:6d}'.format(iternum), '   time: {:.2f}'.format(stopt-self.startt), file=output)
            print('Func: {:.2e}'.format(self.Fs[-1]), ' |g|: {:.2e}'.format(self.Gs[-1]), file=output)
        self.startt = stopt
        return None
    def __call__(self, xopt, **args):
        if self.ITERNUM%self.recordcycle == 0:
            self.record(xopt, iternum=self.ITERNUM, **args)
        if self.ITERNUM%self.savecycle == 0:
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
    # 引入max_f_prime
    max_f_prime = 0.1
    linpdelearner = pdelearner.VariantCoeLinear1d(T=namestobeupdate['T'], N=namestobeupdate['N'], X=namestobeupdate['X'],
                                                  batch_size=namestobeupdate['batch_size'], u0=u_0, dt=namestobeupdate['dt'], time_steps=namestobeupdate['time_steps'],
                                                  dx=namestobeupdate['dx'], max_f_prime=max_f_prime, u_fixed=u_fixed, theta=namestobeupdate['theta'],
                                                  device=namestobeupdate['device'], is_train=True)

    if namestobeupdate['precision'] == 'double':
        linpdelearner.double()
    else:
        linpdelearner.float()

    linpdelearner.to(namestobeupdate['device'])
    callback = callbackgen(options)  # some useful interface
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
        loss = loss+((p<s).to(p)*0.5/s*p**2).sum()+((p>=s).to(p)*(p-s/2)).sum()
    return loss


def print_model_parameters(linpdelearner):
    for parameters in linpdelearner.parameters():
        print(parameters)

def printcoeffs(linpdelearner):
    for poly in linpdelearner.polys:
        tsym_0, csym_0, tsym_1, csym_1 = poly.coeffs()
        str_molecular = '(' + str(csym_0[0]) + ')' + '*' + str(tsym_0[0])
        for index in range(1, len(tsym_0)):
            str_molecular += '+' + '(' + str(csym_0[index]) + ')' + '*' + str(tsym_0[index])
        str_denominator = '(' + str(csym_1[0]) + ')' + '*' + str(tsym_1[0])
        for index in range(1, len(tsym_1)):
            str_denominator += '+' + '(' + str(csym_1[index]) + ')' + '*' + str(tsym_1[index])
        print(str_molecular)
        print(str_denominator)


# 修改成按固定时间点的loss
def loss(model, stepnum, obs_data, layerweight=None):
    # 注意这个地方 stepnum 和 model.time_steps 是不一样的
    if layerweight is None:
        layerweight = [1,]*stepnum
        layerweight[-1] = 1
    ut = model.u0
    stableloss = 0
    dataloss = 0
    sparseloss = 0
    # sparseloss = _sparse_loss(model)
    # 模型更新
    model.update()
    # 调试：打印优化过程中的模型
    # print_model_parameters(model)
    printcoeffs(model)

    # 时间list
    obs_t = []
    for i in range(stepnum):
        obs_t.append(0.1 * i)
    # 真实数据， 步骤list
    dt_fixed = 0.004425
    obs_time_step = []

    for ele in obs_t:
        obs_time_step.append(round(ele/dt_fixed))
    obs_data_choose = obs_data[obs_time_step, :, :]

    # 预测数据， 步骤list
    dt_changed = model.dt.item()
    pre_time_step = []
    for ele in obs_t:
        pre_time_step.append(round(ele/dt_changed))
    # 预测的轨迹
    trajectories = model(ut, pre_time_step[-1] + 1)
    pre_data_choose = trajectories[pre_time_step, :, :]

    # 各种loss的计算
    du = 1.0/500
    u_fixed_0 = 0.0
    u_fixed_np = np.zeros((1, 501), dtype=float)
    u_fixed_np[:1, 0] = u_fixed_0
    for i in range(1, 501):
        u_fixed_np[:1, i] = u_fixed_0 + i * du
    u_fixed = torch.from_numpy(u_fixed_np)
    u_fixed = u_fixed.to(model.device)

    u_fixed.requires_grad = True
    f_test = model.f_predict(u_fixed)
    dfdu = grad(f_test, u_fixed, grad_outputs=torch.ones_like(f_test), create_graph=False)[0]
    u_fixed.requires_grad = False
    # max_f_prime = torch.max(torch.abs(dfdu))

    # 自己写的导数部分
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
        dfdu_1[:1, i] = ((f_test_1[:, i+1] - f_test_1[:, i])/du)
    dfdu_1 = dfdu_1.to(dtype=torch.float64)
    max_f_prime = torch.max(torch.abs(dfdu_1))

    # 检测模型除法的部分
    u_fixed = u_fixed.unsqueeze(1)
    outputs_deno = u_fixed.permute(0, 2, 1)
    for k in range(model.poly0.hidden_layers):
        # id
        id_list = []
        for index in range(outputs_deno.shape[2]):
            id_list.append(outputs_deno[:, :, index:index+1])
        id = torch.cat(id_list, dim=-1)
        # multiply
        o = model.poly0.layer[k](outputs_deno)
        outputs_deno = torch.cat([id, o[..., :1] * o[..., 1:]], dim=-1)
    deno = model.poly0.denominator(outputs_deno)
    zero = torch.zeros_like(deno)
    penalty = torch.abs(torch.where(deno > model.theta, zero, deno)).sum()

    # 打印相关的数据
    print('obs_time_step:')
    print(obs_time_step)
    print("dt")
    print(dt_changed)
    print('pre_time_step:')
    print(pre_time_step)
    print('max_f_prime:')
    print(model.max_f_prime)

    # zero = torch.zeros_like(dfdu_1)
    # max_f_prime_loss = torch.where(torch.abs(dfdu_1) < 100, zero, torch.abs(dfdu_1)).sum()
    # max_f_prime_loss = torch.max(torch.abs(dfdu_1))

    print(dfdu[0, 0:10])
    print(dfdu_1[0, 0:10])

    loss = 0
    max_f_prime_loss = 0
    if 0 < max_f_prime and max_f_prime < 100:
        dataloss = mse(obs_data_choose[:, :, :], pre_data_choose[:, :, :])
        loss = dataloss
    elif max_f_prime >= 100 and penalty > 0:
        loss = penalty
    else:
        print("\033[32mmax_f_prime %.6f, \033[0m" % (max_f_prime))
        max_f_prime_loss = max_f_prime - 100
        loss = max_f_prime_loss
    # loss = dataloss + penalty + max_f_prime_loss
    print("\033[33mloss0 %.6f, data loss0 %.6f, max_f_prime_loss0 %.6f, stable loss %.6f, sparse loss %.6f, max_f_prime %.6f, penalty %.6f, \033[0m" % (loss, dataloss, max_f_prime_loss, 0.05*stableloss, 0.005*sparseloss, max_f_prime, penalty)) # 黄色
    return loss, sparseloss, stableloss

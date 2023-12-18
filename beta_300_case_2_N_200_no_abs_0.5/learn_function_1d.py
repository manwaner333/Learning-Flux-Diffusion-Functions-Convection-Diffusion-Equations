#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
import torch
from aTEAM.optim import NumpyFunctionInterface
import linpdeconfig
np.random.seed(0)
torch.manual_seed(0)
# from torchviz import make_dot

options = {
    '--device': 'cpu',  # 'cuda:0',
    '--precision': 'double',
    '--taskdescriptor': 'beta_300_case_2_N_200_0.5',
    '--batch_size': 6,  # 250,  # 150,
    '--maxiter': 100,   # 156,
    '--X': 10,
    '--T': 2.0,
    '--dx': 0.05,   # 0.05,  # 0.025, # 0.05
    '--N': 200,   # 200  # cell number
    '--N_a': 400,   # 200  # cell number
    '--dt': 0.169492,
    '--time_steps': 236,
    '--layer': 10,
    '--recordfile': 'convergence',
    '--recordcycle': 10,
    '--savecycle': 10000,
    '--theta': 0.001
}

options = linpdeconfig.setoptions(argv=sys.argv[1:], kw=options, configfile=None)
namestobeupdate, callback, linpdelearner = linpdeconfig.setenv(options)
globals().update(namestobeupdate)

# init parameters
def initexpr(model):
    rhi = model.polys
    for poly in rhi:
        for p in poly.parameters():
            p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device) * 1e-1 * 6
    spm1 = model.diffs1
    for diff in spm1:
        for p in diff.parameters():
            p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device) * 1e-1 * 3
    spm2 = model.diffs2
    for diff in spm2:
        for p in diff.parameters():
            p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device) * 1e-1 * 3
    spm3 = model.diffs3
    for diff in spm3:
        for p in diff.parameters():
            p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device) * 1e-1 * 6
    spm4 = model.diffs4
    for diff in spm4:
        for p in diff.parameters():
            p.data = torch.randn(*p.shape, dtype=p.dtype, device=p.device) * 1e-1 * 6
    return None
initexpr(linpdelearner)


for name, parameters in linpdelearner.named_parameters():
    print(name, ':', parameters)


params = list(linpdelearner.coe_params())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))


for l in [layer]:
    callback.stage = 'layer-'+str(l)
    isfrozen = False
    stepnum = (l if l>=1 else 1)
    # load real data
    real_data_file = 'data/' + taskdescriptor + '_U' + '.npy'
    obs_data = torch.from_numpy(np.load(real_data_file))
    obs_data = obs_data.to(device)

    def forward():
        sparsity = 0.00
        stablize = 0.00
        dataloss, sparseloss, stableloss = linpdeconfig.loss(linpdelearner, stepnum, obs_data)
        loss = dataloss + sparsity * sparseloss + stablize * stableloss
        return loss

    nfi = NumpyFunctionInterface(
        [dict(params=linpdelearner.coe_params(), isfrozen=False, x_proj=None, grad_proj=None)],
        forward=forward, always_refresh=False)
    callback.nfi = nfi


    xopt, f, d = lbfgsb(nfi.f, nfi.flat_param, nfi.fprime, m=500, callback=callback, factr=1e0, pgtol=1e-16, maxiter=maxiter, iprint=1,)
    nfi.flat_param = xopt
    callback.save(xopt, 'final')

    for name, parameters in linpdelearner.named_parameters():
        print(name, ':', parameters)

    def printcoeffs():
        with callback.open() as output:
            print('current expression:', file=output)
            for poly in linpdelearner.polys:
                tsym_0, csym_0, tsym_1, csym_1 = poly.coeffs()
                print(tsym_0[:20], file=output)
                print(csym_0[:20], file=output)
                print(tsym_1[:20], file=output)
                print(csym_1[:20], file=output)
                str_molecular = '(' + str(csym_0[0]) + ')' + '*' + str(tsym_0[0])
                for index in range(1, len(tsym_0)):
                    str_molecular += '+' + '(' + str(csym_0[index]) + ')' + '*' + str(tsym_0[index])

                str_denominator = '(' + str(csym_1[0]) + ')' + '*' + str(tsym_1[0])
                for index in range(1, len(tsym_1)):
                    str_denominator += '+' + '(' + str(csym_1[index]) + ')' + '*' + str(tsym_1[index])
                print(str_molecular)
                print(str_denominator)

    printcoeffs()
    print(d['warnflag'])
    print(d['task'])



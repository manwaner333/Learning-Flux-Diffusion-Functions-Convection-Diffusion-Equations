"""
3d examples for LagrangeInterp,LagrangeInterpFixInputs (nn.modules.Interpolation)
"""
#%%
from numpy import *
import numpy as np
import torch
from torch.autograd import Variable,grad
import torch.nn as nn
from torch.nn import functional as F
from scipy.optimize.lbfgsb import fmin_l_bfgs_b as lbfgsb
from scipy.optimize.slsqp import fmin_slsqp as slsqp
import matplotlib.pyplot as plt
from aTEAM.optim import NumpyFunctionInterface,ParamGroupsManager
from aTEAM.nn.modules import LagrangeInterp,LagrangeInterpFixInputs
from aTEAM.utils import meshgen
#%%
def testfunc(inputs):
    """inputs (ndarray)"""
    return sin(inputs[...,0]*8)+cos(sqrt(inputs[...,1]*4))*sin(inputs[...,2]*4)
def compare(I, inputs):
    infe = I(inputs).data.cpu().numpy()
    infe_true = testfunc(inputs.data.cpu().numpy())
    return infe,infe_true
def forward(I, inputs):
    outputs = I(inputs)
    outputs_true = torch.from_numpy(testfunc(inputs.data.cpu().numpy()))
    outputs_true = outputs.data.new(outputs_true.size()).copy_(outputs_true)
    outputs_true = Variable(outputs_true)
    return ((outputs-outputs_true)**2).mean()
def forwardFixInputs(IFixInputs, outputs_true):
    outputs = IFixInputs()
    return ((outputs-outputs_true)**2).mean()
#%%
m = 3
d = 2
device = -1
mesh_bound = zeros((2,m))
# mesh_bound[0] = arange(m)-1
# mesh_bound[1] = arange(m)+1
mesh_bound[0] = 0
mesh_bound[1] = 1
mesh_size = array([40,]*m)
I = LagrangeInterp(m, d, mesh_bound, mesh_size)
I.double()
if device>=0:
    I.cuda(device)
mesh_bound[1] += 1/200
dataset = meshgen(mesh_bound, [201,201,201])
dataset = torch.from_numpy(dataset).clone()
dataset = I.interp_coe.data.new(dataset.size()).copy_(dataset)
dataset = Variable(dataset)
mesh_bound[1] -= 1/200
IFixInputs = LagrangeInterpFixInputs(dataset[:1,:1,:1],m,d,mesh_bound,mesh_size)
IFixInputs.double()
if device>=0:
    IFixInputs.cuda(device)
#%%
inputs_shape = [50,50,50]
IN,JN,KN = int(200/inputs_shape[0]), int(200/inputs_shape[1]), int(200/inputs_shape[2])
indx = zeros((IN*JN*KN,3),dtype=int32)
idx = 0
for i in range(IN):
    for j in range(JN):
        for k in range(KN):
            indx[idx] = array([i,j,k])*array(inputs_shape)
            idx += 1
#%%
nfi = NumpyFunctionInterface([I.interp_coe,],forward=lambda :forward(I,dataset))
nfi.flat_param = random.randn(nfi.numel())
x0 = nfi.flat_param
for i in range(64):
    inputs = dataset[
            indx[i,0]:indx[i,0]+inputs_shape[0],
            indx[i,1]:indx[i,1]+inputs_shape[1],
            indx[i,2]:indx[i,2]+inputs_shape[2]
            ]
    inputs = inputs.clone()
    nfi.forward = lambda :forward(I,inputs)
    x = nfi.flat_param
    x,f,d = lbfgsb(nfi.f,x,nfi.fprime,m=1000,maxiter=20,factr=1,pgtol=1e-16,iprint=10)
#%%
outputs = IFixInputs()
outputs_true = torch.from_numpy(testfunc(IFixInputs.inputs.cpu().numpy()))
outputs_true = outputs_true.view(outputs.size())
outputs_true = outputs.data.new(outputs_true.size()).copy_(outputs_true)
outputs_true = Variable(outputs_true)

nfi = NumpyFunctionInterface([IFixInputs.interp_coe,],forward=lambda :forwardFixInputs(IFixInputs,outputs_true))
nfi.flat_param = random.randn(nfi.numel())
for i in range(64):
    inputs = dataset[
            indx[i,0]:indx[i,0]+inputs_shape[0],
            indx[i,1]:indx[i,1]+inputs_shape[1],
            indx[i,2]:indx[i,2]+inputs_shape[2]
            ]
    inputs = inputs.clone()
    IFixInputs.inputs = inputs
    outputs = IFixInputs()
    outputs_true = torch.from_numpy(testfunc(IFixInputs.inputs.cpu().numpy()))
    outputs_true = outputs_true.view(outputs.size())
    outputs_true = outputs.data.new(outputs_true.size()).copy_(outputs_true)
    outputs_true = Variable(outputs_true)
    nfi.forward = lambda :forwardFixInputs(IFixInputs,outputs_true)
    x = nfi.flat_param
    x,f,d = lbfgsb(nfi.f,nfi.flat_param,nfi.fprime,m=1000,maxiter=20,factr=1,pgtol=1e-14,iprint=10)
#%%
inputs = dataset[
        random.randint(200/inputs_shape[0])+int(200/inputs_shape[0])*arange(0,inputs_shape[0],dtype=int32)[:,newaxis,newaxis],
        random.randint(200/inputs_shape[1])+int(200/inputs_shape[1])*arange(0,inputs_shape[1],dtype=int32)[newaxis,:,newaxis],
        random.randint(200/inputs_shape[2])+int(200/inputs_shape[2])*arange(0,inputs_shape[2],dtype=int32)[newaxis,newaxis,:]
        ]
inputs = inputs.clone()
nfi.forward = lambda :forward(I,inputs)
infe,infe_true = compare(I,inputs)
print(sqrt((infe-infe_true)**2).mean())
print(sqrt((infe-infe_true)**2).max())
h = plt.figure()
indx = random.randint(20)
a = h.add_subplot(4,2,1)
a.imshow(infe_true[indx])
a.set_title('true')
a = h.add_subplot(4,2,2)
a.imshow(infe[indx])
a.set_title('inferenced')
indx = random.randint(20)
a = h.add_subplot(4,2,3)
a.plot(infe_true[indx,indx])
a = h.add_subplot(4,2,4)
a.plot(infe[indx,indx])
indx = random.randint(20)
a = h.add_subplot(4,2,5)
a.plot(infe_true[indx,:,indx])
a = h.add_subplot(4,2,6)
a.plot(infe[indx,:,indx])
indx = random.randint(20)
a = h.add_subplot(4,2,7)
a.plot(infe_true[:,indx,indx])
a = h.add_subplot(4,2,8)
a.plot(infe[:,indx,indx])
#%%
inputs = dataset[
        random.randint(200/inputs_shape[0])+int(200/inputs_shape[0])*arange(0,inputs_shape[0],dtype=int32)[:,newaxis,newaxis],
        random.randint(200/inputs_shape[1])+int(200/inputs_shape[1])*arange(0,inputs_shape[1],dtype=int32)[newaxis,:,newaxis],
        random.randint(200/inputs_shape[2])+int(200/inputs_shape[2])*arange(0,inputs_shape[2],dtype=int32)[newaxis,newaxis,:]
        ]
inputs = inputs.clone()
IFixInputs.inputs = inputs
outputs = IFixInputs()
outputs_true = torch.from_numpy(testfunc(IFixInputs.inputs.cpu().numpy()))
outputs_true = outputs_true.view(outputs.size())
infe = outputs.data.cpu().numpy()
infe_true = outputs_true.numpy()
print(sqrt((infe-infe_true)**2).mean())
print(sqrt((infe-infe_true)**2).max())
h = plt.figure()
indx = random.randint(20)
a = h.add_subplot(4,2,1)
a.imshow(infe_true[indx])
a.set_title('true')
a = h.add_subplot(4,2,2)
a.imshow(infe[indx])
a.set_title('inferenced')
indx = random.randint(20)
a = h.add_subplot(4,2,3)
a.plot(infe_true[indx,indx])
a = h.add_subplot(4,2,4)
a.plot(infe[indx,indx])
indx = random.randint(20)
a = h.add_subplot(4,2,5)
a.plot(infe_true[indx,:,indx])
a = h.add_subplot(4,2,6)
a.plot(infe[indx,:,indx])
indx = random.randint(20)
a = h.add_subplot(4,2,7)
a.plot(infe_true[:,indx,indx])
a = h.add_subplot(4,2,8)
a.plot(infe[:,indx,indx])


#%%

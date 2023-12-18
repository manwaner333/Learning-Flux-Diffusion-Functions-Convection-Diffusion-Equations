import numpy as np
import torch
import linpdeconfig

# 加载模型
experiment_name = 'dx_0.05_example_1_layer_20_upper_10_cpu_jump_1'
configfile = 'checkpoint/' + experiment_name + '/options.yaml'
train_layers = 20
options = linpdeconfig.setoptions(configfile=configfile, isload=True)
namestobeupdate, callback, linpdelearner = linpdeconfig.setenv(options)
globals().update(namestobeupdate)
callback.load(train_layers)
model = linpdelearner

# 给模型重新赋一些变量的值: u0,
model.batch_size = 1
u_0_np = np.zeros((1, namestobeupdate['N']), dtype=float)
u_0_np[:1, 80:120] = 0.8
u_0 = torch.from_numpy(u_0_np)
u_0 = u_0.to(namestobeupdate['device'])
model.u0 = u_0
max_f_prime = torch.DoubleTensor(1).fill_(-0.1)
model.max_f_prime = max_f_prime


# 预测
test_time_steps = 500
trajectories = model(u_0, test_time_steps)
print("new time_steps")
print(model.time_steps)
print("max value of prediction data:")
print(torch.max(trajectories[499, :, :]))

# 保存预测数据
test_data_name = experiment_name
np.save('data/' + 'predict_' + test_data_name + '.npy', trajectories.data)






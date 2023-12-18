# aTEAM
A pyTorch Extension for Applied Mathematics

This version is compatible with pytorch (0.3.1). You can create a conda environment for pytorch0.3.1 if you have a newer pytorch release in your base env:
```
conda create -n torch0.3 python=3 jupyter
source activate torch0.3
pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
# change ".../cu90/..." to ".../cpu/..." or ".../cu91/..." if needed
wget https://github.com/ZichaoLong/aTEAM/archive/v0.1.tar.gz
tar -xf v0.1.tar.gz
```

## Some code maybe useful to you:

- aTEAM.optim.NumpyFuntionInterface: This function enable us to optimize pytorch modules with external optimizer such as scipy.optimize.lbfgsb.fmin_l_bfgs_b
- aTEAM.nn.functional.utils.tensordot: It is similar to numpy.tensordot
- aTEAM.nn.modules.MK: [Moment matrix](https://arxiv.org/abs/1710.09668) & convolution kernel convertor
- ...

For more usages pls refer to aTEAM/test/*.py

# PDE-Net

Initially, aTEAM is written for this paper:

[PDE-Net: Learning PDEs from Data](https://arxiv.org/abs/1710.09668)[(ICML 2018)](https://icml.cc/Conferences/2018)[(source code)](https://github.com/ZichaoLong/PDE-Net)<br />
[Long Zichao](http://zlong.me/), [Lu Yiping](http://about.2prime.cn/), [Ma Xianzhong](https://www.researchgate.net/profile/Xianzhong_Ma), [Dong Bin](http://bicmr.pku.edu.cn/~dongbin)

If you find this code useful in your research then please cite
```
@inproceedings{long2017pde,
    title={PDE-Net: Learning PDEs from Data},
    author={Long, Zichao and Lu, Yiping and Ma, Xianzhong and Dong, Bin},
    booktitle={Proceedings of the 35th International Conference on Machine Learning (ICML 2018)},
    year={2018}
}
```

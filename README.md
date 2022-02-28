# Model-based Deep 3D holographic imaging

This is the open source repository for our paper of: ["Holographic 3D particle imaging with model-based deep network"](https://ieeexplore.ieee.org/document/9369862), [Ni Chen](https://ni-chen.github.io/), [Congli Wang](https://congliwang.github.io/), and [Wolfgang Heidrich](https://vccimaging.org/People/heidriw/), [IEEE Transactions on Computational Imaging](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6745852) 7: 288-296, 2021.


--------------------------------------------

### Overview

This repository contains:

- Keras implementation of the MB-HoloNet
- MATLAB implementation of inline hologram synthesize



----------------------------------

### Requirements

 - Python 3.6.5 
 - Tensorflow 1.15 
 - Keras 2.2.5 




## How to use the code

- If you have your own data, put it in `./code/data/`, then run `./code/keras/train.py` after setting your parameters
- If you don't have a code, `./code/matlab/HoloData_sim.m` can generate training data.
- You can also down a sample data here: https://drive.google.com/file/d/10i2N8HV9x5AOxZZ5OVOke0LgYENZKZ1E/view?usp=sharing, put it in  `./code/data/`, and then run the `./code/keras/train.py`



-------------------------------------

## Contact

Ni Chen, ni.chen@kaust.edu.sa


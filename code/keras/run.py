# %% run tensorboard in ibex
'''
1. go into the working dir
cd VolHolography
conda activate tf-gpu
tensorboard --logdir=/home/chenn/VolHolography/keras/logs --port 6006
# tensorboard --logdir=/home/wangc0a/chen/keras/logs --port 6006

If port is unavailable
lsof -ti:16006 | xargs kill -9

ssh -L localhost:16006:localhost:6006 chenn@glogin.ibex.kaust.edu.sa
ssh -L localhost:16007:localhost:6021 wangc0a@10.109.66.157
ssh -L localhost:16008:localhost:6008 wangc0a@glogin.ibex.kaust.edu.sa

2. Set SSH tunnel
'''

# %% tensorboard
# tensorboard --logdir=logs --host localhost --port 6021


# %%

import os

from utils import *



# %% simulation


# #################################################### Nz=7 ####################################################
os.system("python train.py --layer_num %d --start_epoch 0  --epochs 1000 --batch_size 32  --lr_max 2e-3 --lr_min 10e-5 --gamma 1e-4 --reg 5e-4 --Nxy 32 --Nz 7 --dz %s --ppv %s --data_num 1000  --obj_type %s"  % (5, '1200um', '1e-03~5e-03', 'sim'))


os.system("python test.py --layer_num %d --start_epoch 0  --epochs 1000 --batch_size 32  --lr_max "
          "2e-3 --lr_min 10e-5 --gamma 1e-4 --reg 5e-4 --Nxy 32 --Nz 7 --dz %s --ppv %s --data_num 1000  --obj_type %s"  % (5, '1200um', '1e-03~5e-03', 'sim'))




# %% windows
import sys

if sys.platform == 'win32':
    import winsound
    winsound.Beep(600, 250)
else:
    os.system('play -nq -t alsa synth {} sine {}'.format(1, 400))

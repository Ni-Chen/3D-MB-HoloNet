# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:46:04 2019
@author: Ni Chen

References:
    - https://blog.csdn.net/jinping_shi/article/details/52433975

Environment:
- Python 3.6.5
- Tensorflow 1.15
- Keras 2.2.5
"""

# %%
from __future__ import print_function

from argparse import ArgumentParser

import scipy.io
from holonet import MOHoloNet
from keras.backend.tensorflow_backend import set_session
from utils import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True
set_session(tf.Session(config=config))  # set this TensorFlow session as the default session for Keras

# %%
parser = ArgumentParser(description='MO-HoloNet')

parser.add_argument('--obj_type', type=str, default='sim', help='exp or sim')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='continue training from an epoch')
parser.add_argument('--epochs', type=int, default=1, help='epochs')

parser.add_argument('--layer_num', type=int, default=1,
                    help='phase number of MO-HoloNet')
parser.add_argument('--lr_max', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_min', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=1e-3, help='loss parameter')
parser.add_argument('--reg', type=float, default=1e-7,
                    help='regularization parameter')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

parser.add_argument('--data_num', type=int, default=10, help='data num')
parser.add_argument('--ppv', type=str, default='5e-03', help='ppv')
parser.add_argument('--Nxy', type=int, default=32, help='lateral size')
parser.add_argument('--Nz', type=int, default=3, help='depth number')
parser.add_argument('--dz', type=str, default='1200um', help='depth interval')

args = parser.parse_args()

sys_param = args.obj_type + '_Nz' + str(args.Nz) + '_ppv' + args.ppv + '_dz' + args.dz + \
            '_L' + str(args.layer_num) + '_B' + str(args.batch_size) + '_N' + str(args.data_num) + \
            '_lr' + str(args.lr_max) + '_G' + str(args.gamma) + \
            '_R' + str(args.reg)

data_dir = '../data/' + args.obj_type + '_Nz' + str(args.Nz) + '_ppv' + args.ppv + '_50db_dz' + args.dz
out_dir = './output/'

model_name = './model/gold/' + 'holonet_' + sys_param

# %% Test

test_dir = data_dir
test_data, test_label, otf3d = load_holo(data_dir + '.mat', args.data_num + 1, 50)

val = scipy.io.loadmat('./model/' + 'holonet_' + sys_param + '.mat')
data_mean = val['data_mean']
data_std = val['data_std']
data_max = val['data_max']
data_min = val['data_min']
print('training data: max:' + str(data_max) + ', min: ' + str(data_std) + ', mean:' + str(
    data_mean) + ', std: ' + str(data_std))
test_data = norm_data_mu(test_data, data_mean, data_std)


print('test data: max:' + str(np.amax(test_data)) + ', min: ' + str(np.amin(test_data)) + ', mean:' + str(np.mean(test_data)) + ', std:' + str(np.std(test_data)))

# val = scipy.io.loadmat('./model/' + 'holonet_' + sys_param + '.mat')
# data_mean = val['data_mean']
# data_std = val['data_std']
# data_max = val['data_max']
# data_min = val['data_min']
# print('training data: max:' + str(data_max) + ', min: ' + str(data_std)+', mean:' + str(data_mean) + ', std: ' + str(data_std))
# test_data = norm_data_mu(test_data, data_mean, data_std)
#
# test_data, no1, no2 = norm_data(test_data)

# test_data = test_data*0.98
# test_dir = data_dir
# test_data, test_label, otf3d = load_holo(data_dir + '.mat', args.data_num + 1, 50)
#
# val = scipy.io.loadmat('./model/' + sys_param + '.mat')
# data_mean = val['data_mean']
# data_std = val['data_std']
# data_max = val['data_max']
# data_min = val['data_min']
# print('training data: max:' + str(data_max) + ', min: ' + str(data_std) + ', mean:' + str(data_mean) + ', std: ' + str(
#     data_std))
# test_data = norm_data_mu(test_data, data_mean, data_std)

#%%
test_num = test_data.shape[0]

print('Num of tests:' + str(test_num))

model = MOHoloNet(args.Nxy, args.Nxy, args.Nz, args.layer_num, args.lr_max, args.gamma, args.reg).build()
model.load_weights(model_name + '.hdf5')

timer = Timer()
test_predict = model.predict([test_data, np.tile(otf3d, (test_num, 1, 1, 1))], verbose=1)
timer.timer()

scipy.io.savemat(out_dir + sys_param + '_data.mat', {'data': test_data})
scipy.io.savemat(out_dir + sys_param + '_predict.mat', {'predict': test_predict})
scipy.io.savemat(out_dir + sys_param + '_gt.mat', {'gt': test_label})

#%%
test_mae = np.mean(np.abs(test_label - test_predict))
print('Evaluate...')
scores = model.evaluate([test_data, np.tile(otf3d, (test_num, 1, 1, 1))], test_label)

print('Eval score of %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
print('Test loss:', test_mae)

for show_idx in [0]:
# for show_idx in range(test_num):
    gt_single = test_label[show_idx, :, :, :]
    pred_single = test_predict[show_idx, :, :, :]

    disp_str = 'N' + str(args.data_num) + '_lr' + str(args.lr_max) + '_G' + str(args.gamma) + '_R' + str(args.reg)
    plotcube(gt_single, "Slice" + str(show_idx+1) + ": GT_" + disp_str, out_dir + sys_param + '_gt.png')
    plotcube(pred_single, "Slice" + str(show_idx+1) + ": Pred_" + disp_str, out_dir + sys_param + '_predict.png')

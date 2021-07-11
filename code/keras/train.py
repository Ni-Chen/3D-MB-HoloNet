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

import datetime
from argparse import ArgumentParser

import scipy.io
from keras.backend.tensorflow_backend import set_session
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler

from holonet import MOHoloNet
from utils import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True
set_session(tf.Session(config=config))  # set this TensorFlow session as the default session for Keras

# %%
parser = ArgumentParser(description='MO-HoloNet')

parser.add_argument('--obj_type', type=str, default='sim', help='exp or sim')
parser.add_argument('--start_epoch', type=int, default=0, help='continue training')
parser.add_argument('--epochs', type=int, default=1000, help='epochs')

parser.add_argument('--layer_num', type=int, default=1,  help='phase number of MO-HoloNet')
parser.add_argument('--lr_max', type=float, default=1e-3, help='learning rate')
parser.add_argument('--lr_min', type=float, default=1e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=1e-4, help='loss parameter')
parser.add_argument('--reg', type=float, default=1e-4, help='regularization parameter')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')

parser.add_argument('--data_num', type=int, default=1000, help='data num')
parser.add_argument('--ppv', type=str, default='5e-03', help='ppv')
parser.add_argument('--Nxy', type=int, default=32, help='lateral size')
parser.add_argument('--Nz', type=int, default=3, help='depth number')
parser.add_argument('--dz', type=str, default='1200um', help='depth interval')

# parser.add_argument('--note', type=str, default='', help='notes')

args = parser.parse_args()

sys_param = args.obj_type + '_Nz' + str(args.Nz) + '_ppv' + args.ppv + '_dz' + args.dz + \
            '_L' + str(args.layer_num) + '_B' + str(args.batch_size) + '_N' + str(args.data_num) + \
            '_lr' + str(args.lr_max) + '_G' + str(args.gamma) + '_R' + str(args.reg)

data_dir = '../data/' + args.obj_type + '_Nz' + str(args.Nz) + '_ppv' + args.ppv + '_50db_dz' + args.dz
out_dir = './output/'

epoch_new = args.start_epoch + args.epochs

model_name = './model/' + 'holonet_' + sys_param


# %%  Training
print('Loading data...')

train_data_all, train_label_all, otf3d = load_holo(data_dir + '.mat', 1, args.data_num)

train_data, val_data, train_label, val_label = train_test_split(train_data_all, train_label_all, test_size=0.2,
                                                                random_state=42)
train_num = train_data.shape[0]

#%%
data_max = np.amax(train_data)
data_min = np.amin(train_data)

train_data, data_mean, data_std = norm_data(train_data)

val = {}
val['data_max'] = data_max
val['data_min'] = data_min
val['data_mean'] = data_mean
val['data_std'] = data_std
scipy.io.savemat('./model/' + sys_param + '.mat', val)
val_data = norm_data_mu(val_data, data_mean, data_std)

# Save data information
print('training data mean:' + str(data_mean) + ', std: ' + str(data_std) + ', max:' + str(data_max) + ', min: ' + str(data_min))


#%%

model = MOHoloNet(args.Nxy, args.Nxy, args.Nz, args.layer_num, args.lr_max, args.gamma, args.reg).build()

if args.start_epoch > 0:
    model.load_weights(model_name + '.hdf5')
    # args.batch_size = args.batch_size / 2
    # args.lr_max = args.lr_max/10
    # args.lr_min = args.lr_min/10

model_checkpoint = ModelCheckpoint(model_name + '.hdf5',
                                   monitor='val_loss',
                                   verbose=2,
                                   save_best_only=True,
                                   mode='auto')

log_dir = './logs/' + datetime.datetime.now().strftime("%m%d-%H%M") + '_' + sys_param
tbCallBack = TensorBoard(log_dir=log_dir,
                         profile_batch=0,  # prevent .profile-empty blocks loading data
                         )

schedule = PolynomialDecay(maxEpochs=args.epochs, initAlpha=args.lr_max, endAlpha=args.lr_min, power=2)  # Linear Decay (p=1)
callbacks = [model_checkpoint, tbCallBack, LearningRateScheduler(schedule)]

train_num = train_data.shape[0]
test_num = val_data.shape[0]

timer = Timer()
hist = model.fit([train_data, np.tile(otf3d, (train_num, 1, 1, 1))], train_label,
                 initial_epoch=args.start_epoch,
                 batch_size=args.batch_size,
                 epochs=args.epochs,
                 verbose=2,
                 validation_data=([val_data, np.tile(otf3d, (test_num, 1, 1, 1))], val_label),
                 callbacks=callbacks
                 )

timer.timer()

#%%   test
test_dir = data_dir
test_data, test_label, otf3d = load_holo(data_dir + '.mat', args.data_num, 50)
test_data = norm_data_mu(test_data, data_mean, data_std)

model = MOHoloNet(args.Nxy, args.Nxy, args.Nz, args.layer_num, args.lr_max, args.gamma, args.reg).build()
model.load_weights(model_name + '.hdf5')
test_num = test_data.shape[0]

timer = Timer()
test_predict = model.predict([test_data, np.tile(otf3d, (test_num, 1, 1, 1))], batch_size=1, verbose=1)
timer.timer()
scipy.io.savemat(out_dir + sys_param + '_predict.mat', {'predict': test_predict})
scipy.io.savemat(out_dir + sys_param + '_gt.mat', {'gt': test_label})

#%%
test_mae = np.mean(np.abs(test_label - test_predict))
scores = model.evaluate([test_data, np.tile(otf3d, (test_num, 1, 1, 1))], test_label)

print('Eval score of %s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))
print('Test loss:', test_mae)
for show_idx in [5, 30]:
    gt_single = test_label[show_idx, :, :, :]
    pred_single = test_predict[show_idx, :, :, :]

    disp_str = 'N' + str(args.data_num) + '_lr' + str(args.lr_max) + '_G' + str(args.gamma) + '_R' + str(args.reg)
    plotcube(gt_single, "GT" + disp_str, out_dir + sys_param + '_gt.png')
    plotcube(pred_single, "Pred" + disp_str, out_dir + sys_param + '_predict.png')
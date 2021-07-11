# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:46:04 2019
@author: Ni Chen
"""

import math
import time

import h5py
# import hdf5storage
import numpy as np
import tensorflow as tf
from keras import backend as K
from matplotlib import pyplot as plt
from tensorflow import reduce_mean, square, fft2d, ifft2d

from tensorflow import multiply, tile, abs
from tensorflow.keras import regularizers as reg
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.layers import Conv2D, Activation, Add, BatchNormalization

# %%-------------------------------------- Loss ------------------------------------------

# def total_loss(loss_constraint,  loss_weight):
def total_loss(loss_constraint):
    def sum_loss(x_true, x_pred):
        loss_discrepancy = reduce_mean(square(x_pred - x_true))

        # loss_discrepancy = tf.losses.huber_loss(x_true, x_pred, delta=0.0001)
        # loss_discrepancy = tf.keras.losses.logcosh(x_true, x_pred)

        return loss_discrepancy + loss_constraint

    return sum_loss


# %%-------------------------------------- Metrics ------------------------------------------

def norm_img(img, NewMin, NewMax):
    return (img - np.amin(img)) * (NewMax - NewMin) / (np.amax(img) - np.amin(img)) + NewMin


def norm_max(img):
    return img / np.amax(img)


def norm_0_1(tensor_data):
    tensor = tf.div(tf.subtract(tensor_data, tf.reduce_min(tensor_data)), tf.subtract(
        tf.reduce_max(tensor_data), tf.reduce_min(tensor_data)))
    return tensor


def norm_data(datasets):
    print('Before: mean: ' + str(np.mean(datasets)) + ', std: ' + str(np.std(datasets)) + ', max: ' + str(
        np.amax(datasets)) + ', min: ' + str(np.amin(datasets)))

    mean_data = np.mean(datasets)
    std_data = np.std(datasets)

    datasets_new = (datasets - mean_data) / std_data

    print('After: mean: ' + str(np.mean(datasets_new)) + ', std: ' + str(np.std(datasets_new)) + ', max: ' + str(
        np.amax(datasets_new)) + ', min: ' + str(np.amin(datasets_new)))

    return datasets_new, mean_data, std_data


def norm_data_mu(datasets, mean_data=0, std_data=0):
    print('Before: mean: ' + str(np.mean(datasets)) + ', std: ' + str(np.std(datasets)) + ', max: ' + str(
        np.amax(datasets)) + ', min: ' + str(np.amin(datasets)))

    datasets_new = (datasets - mean_data) / std_data

    print('After: mean: ' + str(np.mean(datasets_new)) + ', std: ' + str(np.std(datasets_new)) + ', max: ' + str(
        np.amax(datasets_new)) + ', min: ' + str(np.amin(datasets_new)))

    return datasets_new


def norm_tensor(img, NewMin, NewMax):
    return (img - tf.reduce_min(img)) * (NewMax - NewMin) / (tf.reduce_max(img) - tf.reduce_min(img)) + NewMin


def r2_score(y_true, y_pred):
    # custom R2-score metrics for keras backend
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def pcc(y_true, y_pred):
    '''
    pearsonr correlation coefficient
    '''
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    mx = tf.math.reduce_mean(y_true)
    my = tf.math.reduce_mean(y_pred)

    xm, ym = y_true - mx, y_pred - my

    r_num = tf.math.reduce_mean(tf.multiply(xm, ym))
    r_den = tf.math.reduce_std(xm) * tf.math.reduce_std(ym)

    return r_num / r_den

def psnr(y_true, y_pred):
    # y_true = norm_tensor(y_true, 0.0, 1.0)
    # y_pred = norm_tensor(y_pred, 0.0, 1.0)

    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

# %%-------------------------------------- Display ------------------------------------------
# Plot a 3D matrix slice by slice
def plotcube(vol, fig_title, file_name):
    maxval = np.amax(vol)
    minval = np.amin(vol)
    vol = (vol - minval) / (maxval - minval)

    Nz, Nx, Ny = np.shape(vol)

    if Nz <= 10:
        img_col_n = Nz
    else:
        img_col_n = math.ceil(np.sqrt(Nz))

    img_row_n = math.ceil(Nz / img_col_n)
    image_height = 5
    fig = plt.figure(figsize=(img_col_n * image_height, image_height * img_row_n + 0.5))
    fig.suptitle(fig_title, y=1)
    img_n = 0
    for iz in range(Nz):
        img_n = img_n + 1
        ax = fig.add_subplot(img_row_n, img_col_n, img_n)
        ax.set_title("z " + str(img_n))
        slice = vol[iz, :, :]
        im = ax.imshow(slice, aspect='equal')

    fig.tight_layout()
    plt.savefig(file_name)
    plt.show()


#%%
class MU(tf.keras.layers.Layer):
    def __init__(self, reg_param=1e-3, **kwargs):
        self.reg_param = reg_param
        super(MU, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='{}_mu'.format(self.name),
                                 shape=(1, 1, 1),
                                 regularizer=reg.l2(l=self.reg_param),
                                 initializer="glorot_normal",
                                 constraint=NonNeg(),
                                 dtype=tf.float32,
                                 trainable=True)
        super(MU, self).build(input_shape)

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()
        mu = tile(self.w, (input_shape[1], input_shape[2], input_shape[3]))

        return mu

    def get_config(self):
        config = super(MU, self).get_config().copy()
        config.update({'reg_param': self.reg_param})

        return config


class SoftThreshold(tf.keras.layers.Layer):
    def __init__(self, reg_param=1e-3, **kwargs):
        self.reg_param = reg_param
        super(SoftThreshold, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(name='{}_tau'.format(self.name),
                                    shape=(1, 1, 1),
                                    regularizer=reg.l2(l=self.reg_param),
                                    initializer="glorot_normal",
                                    constraint=NonNeg(),
                                    dtype=tf.float32,
                                    trainable=True)
        super(SoftThreshold, self).build(input_shape)

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()
        bias_tile = tile(self.bias, (input_shape[1], input_shape[2], input_shape[3]))

        # return tf.multiply(tf.sign(inputs), tf.maximum(tf.abs(inputs) - bias_tile, 0))
        return multiply(tf.sign(inputs), tf.nn.relu(abs(inputs) - bias_tile))

    def get_config(self):
        config = super(SoftThreshold, self).get_config().copy()
        config.update({'reg_param': self.reg_param})

        return config


def res_block(X, filters, reg_param):
    init = 'glorot_normal'  # 'random_normal', lecun_normal, glorot_normal, glorot_uniform, glorot_uniform(seed=0)

    X_shortcut = X

    X = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', data_format='channels_first', use_bias=False,
               kernel_regularizer=reg.l2(l=reg_param),
               kernel_initializer=init)(X)
    X = BatchNormalization(axis=1)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', data_format='channels_first', use_bias=False,
               kernel_regularizer=reg.l2(l=reg_param),
               kernel_initializer=init)(X)
    X = BatchNormalization(axis=1)(X)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

# %%-------------------------------------- Data Process ------------------------------------------
# def load_holo(data_dir="./data/", start_index=1, data_num=1):
#     mat = h5py.File(data_dir, "r")
#     data = np.transpose(mat['data'][()]).astype(np.float32)
#
#     label = np.transpose(mat['label'][()]).astype(np.float32)
#     label = np.transpose(label, (0, 3, 1, 2))
#
#     otf3d = np.transpose(mat['otf3d'][()]).view(np.complex)
#     otf3d = np.transpose(otf3d, (2, 0, 1))
#     otf3d = np.expand_dims(otf3d, axis=0)
#
#     data = data[start_index - 1:start_index + data_num - 1, ...]
#     label = label[start_index - 1:start_index + data_num - 1, ...]
#
#     return data, label, otf3d

# def load_holo(data_dir="./data/", start_index=1, data_num=1):
def load_holo(data_dir="./data/", *args):
    n_arg = len(args)

    mat = h5py.File(data_dir, "r")
    data = np.transpose(mat['data'][()]).astype(np.float32)

    label = np.transpose(mat['label'][()]).astype(np.float32)
    label = np.transpose(label, (0, 3, 1, 2))

    otf3d = np.transpose(mat['otf3d'][()]).view(np.complex)
    otf3d = np.transpose(otf3d, (2, 0, 1))
    otf3d = np.expand_dims(otf3d, axis=0)

    if n_arg == 0:
        return data, label, otf3d
    else:
        start_index = args[0]
        data_num = args[1]
        print(start_index)
        print(data_num)
        data = data[start_index - 1:start_index + data_num - 1, ...]
        label = label[start_index - 1:start_index + data_num - 1, ...]

        return data, label, otf3d

#
# %%----------------------------------------------------------------------------
def fftshift2d(a_tensor):
    input_shape = a_tensor.shape.as_list()
    numel = len(input_shape)
    new_tensor = a_tensor
    for axis in range(numel - 2, numel):
        split = (input_shape[axis] + 1) // 2
        mylist = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def ifftshift2d(a_tensor):
    input_shape = a_tensor.shape.as_list()
    numel = len(input_shape)

    new_tensor = a_tensor
    for axis in range(numel - 2, numel):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor


def FT2d(a_tensor):
    return ifftshift2d(fft2d(fftshift2d(a_tensor)))


def iFT2d(a_tensor):
    return ifftshift2d(ifft2d(fftshift2d(a_tensor)))


# %%-----------------------------------------------------------------------------------

def get_lr(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr

# %%  ------------------------------------------- LR realted -----------------------------------------------------------

class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        # compute the set of learning rates for each corresponding epoch
        lrs = [self(i) for i in epochs]


class StepDecay(LearningRateDecay):
    def __init__(self, initAlpha=0.01, factor=0.9, dropEvery=200):
        # store the base initial learning rate, drop factor, and epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        return float(alpha)

class PolynomialDecay(LearningRateDecay):
    def __init__(self, maxEpochs=100, initAlpha=0.01, endAlpha=1e-6, power=1.0):
        # store the maximum number of epochs, base learning rate, and power of the polynomial
        self.maxEpochs = maxEpochs
        self.initAlpha = initAlpha
        self.endAlpha = endAlpha
        self.power = power

    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.maxEpochs))) ** self.power
        # decay = (1 - ((epoch//self.maxEpochs) / float(self.maxEpochs))) ** self.power
        alpha = (self.initAlpha - self.endAlpha) * decay + self.endAlpha

        # return the new learning rate
        return float(alpha)


# %%-----------------------------------------------------------------------------------
class Timer(object):
    def __init__(self):
        self.start_time = time.time()

    def elapsed(self, sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def timer(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time))

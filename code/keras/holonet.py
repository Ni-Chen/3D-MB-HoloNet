# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:46:04 2019
@author: Ni Chen

Setting:
- Python 3.6.5
- Tensorflow 1.15
- Keras 2.2.5
"""

# %%
from __future__ import print_function

from tensorflow import conj, real, subtract, add, divide, cast
from tensorflow.keras.layers import Lambda, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import *

# %%
class MOHoloNet():
    def __init__(self, img_rows=64, img_cols=64, img_depths=5, layer_num=9, lr=1e-3,
                 loss_sym_param=2e-3, reg_param=1e-5):
        self.img_rows = img_rows  # volume width
        self.img_cols = img_cols  # volume height
        self.img_depths = img_depths  # volume depth

        self.phase_number = layer_num

        self.loss_sym_param = loss_sym_param
        self.reg_param = reg_param
        self.lr = lr

        self.filter_num = img_depths

    def backward_wave_prop(self, holo, otf3d):
        '''
        Backward propagation. Propagate a 2D hologram to a 3D volume.
        '''

        Nz = self.img_depths

        holo = Lambda(lambda x: tf.cast(x, tf.complex64))(holo)
        otf3d = Lambda(lambda x: tf.cast(x, tf.complex64))(otf3d)

        conj_otf3d = Lambda(lambda x: conj(x))(otf3d)

        # Perform iFT(FT(O)conj(OTF))
        holo_expand = Lambda(lambda x: tf.expand_dims(x, 1))(holo)
        holo_expand = Lambda(lambda x: tf.tile(x, (1, Nz, 1, 1)))(holo_expand)

        holo_expand = Lambda(lambda x: fft2d(x))(holo_expand)
        field3d_ft = Lambda(lambda x: multiply(x[0], x[1]))([holo_expand, conj_otf3d])
        field3d = Lambda(lambda x: ifft2d(x))(field3d_ft)

        # real constraint
        vol = Lambda(lambda x: real(x))(field3d)

        return vol

    def single_phase(self, v, I, otf3d):
        otf3d = Lambda(lambda x: cast(x, tf.complex64))(otf3d)

        ########################################################################
        mu = MU(self.reg_param)(v)

        # numerator:  b = F( alpha * At * I_h + v)
        o_temp = self.backward_wave_prop(I, otf3d)
        o_temp = Lambda(lambda x: multiply(x[0], x[1]))([mu, o_temp])
        o_add_v = Lambda(lambda x: add(x[0], x[1]))([o_temp, v])
        b = Lambda(lambda x: cast(x, tf.complex64))(o_add_v)
        numerator = Lambda(lambda x: FT2d(x))(b)

        # denominator =FT(b) / (|OTF|^2 + 1)
        otf_square = Lambda(lambda x: real(square(abs(x))))(otf3d)
        otf_square = Lambda(lambda x: multiply(x[0], x[1]))([mu, otf_square])
        one_array = Lambda(lambda x: tf.ones_like(x))(otf_square)
        denominator = Lambda(lambda x: add(x[0], x[1]))([otf_square, one_array])
        denominator = Lambda(lambda x: cast(x, tf.complex64))(denominator)

        x_prime = Lambda(lambda x: divide(x[0], x[1]))([numerator, denominator])
        v_next = Lambda(lambda x: real(iFT2d(x)))(x_prime)

        ########################################################################
        # Proximal: min_o 0.5 ||x -r||_2^2 + theta ||x||_1
        F = res_block
        G = res_block

        o_forward = F(v_next, self.filter_num, self.reg_param)
        o_soft = SoftThreshold(self.reg_param)(o_forward)
        o_next = G(o_soft, self.filter_num, self.reg_param)

        o_forward_backward = G(o_forward, self.filter_num, self.reg_param)
        stage_symloss = Lambda(lambda x: subtract(x[0], x[1]))([o_forward_backward, v_next])

        return o_next, stage_symloss

    def build(self):
        holo_data = Input(shape=(self.img_rows, self.img_cols), dtype=tf.float32)
        otf_data = Input(shape=(self.img_depths, self.img_rows, self.img_cols), dtype=tf.complex64)

        # initial guess
        v_current = self.backward_wave_prop(holo_data, otf_data)

        loss_constraint = 0.0
        for i in range(self.phase_number):
            v_current, stage_symloss = self.single_phase(v_current, holo_data, otf_data)
            loss_constraint += reduce_mean(square(stage_symloss))

        loss_constraint = self.loss_sym_param * loss_constraint / self.phase_number

        # Transfer function to absorption function
        v_current = BatchNormalization(axis=1)(v_current)
        v_current = Activation('sigmoid')(v_current)
        # v_current = Lambda(lambda x: norm_0_1(abs(x)))(v_current)

        model = Model(inputs=[holo_data, otf_data], outputs=v_current)

        optimizer = Adam(lr=self.lr)
        lr_hist = get_lr(optimizer)
        model.compile(
            optimizer=optimizer,
            loss=total_loss(loss_constraint),
            metrics=['acc', 'mae', r2_score, psnr, pcc, lr_hist]
        )
        # model.summary()

        return model

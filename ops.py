"""
StarGAN v2 TensorFlow Implementation
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import tensorflow as tf
import tensorflow_addons as tfa
"""=====================  Initialization ========================"""
weight_initializer = tf.initializers.VarianceScaling(
                        2, 'fan_in','untruncated_normal')
weight_regularizer = tf.keras.regularizers.l2(1e-4) # 1e-4 --> 1e-2
weight_regularizer_fully = tf.keras.regularizers.l2(1e-4) # 1e-4 --> 1e-2

"""========================  Layers  ============================"""
def InstanceNorm(name='InstanceNorm'):
    return tfa.layers.normalizations.InstanceNormalization(
        epsilon=1e-5, scale=True, center=True, name=name)

class Conv(tf.keras.layers.Layer):
    def __init__(self, channels, kernel=3, stride=1, pad=0,
                 pad_type='zero', use_bias=True, sn=False, name='Conv'):
        super(Conv, self).__init__(name=name)
        self.channels = channels
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.conv = SpectralNormalization(tf.keras.layers.Conv2D(
                        filters=self.channels, kernel_size=self.kernel,
                        kernel_initializer=weight_initializer,
                        kernel_regularizer=weight_regularizer,
                        strides=self.stride, use_bias=self.use_bias),
                                              name='sn_' + self.name)
        else:
            self.conv = tf.keras.layers.Conv2D(
                        filters=self.channels, kernel_size=self.kernel,
                        kernel_initializer=weight_initializer,
                        kernel_regularizer=weight_regularizer,
                        strides=self.stride, use_bias=self.use_bias,
                        name = self.name)

    def call(self, x, training=None, mask=None):
        # padding='SAME' ======> pad = floor[ (kernel - stride) / 2 ]
        if self.pad > 0:
            h = x.shape[1]
            if h % self.stride == 0:
                pad = self.pad * 2
            else:
                pad = max(self.kernel - (h % self.stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if self.pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom],
                               [pad_left, pad_right], [0, 0]],
                           mode='REFLECT')
            else:
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom],
                               [pad_left, pad_right], [0, 0]])
        x = self.conv(x)
        return x

    def get_config(self):
        config = super(Conv, self).get_config()
        config.update({"channels": self.channels,
                       'kernel':self.kernel,
                       'stride': self.stride,
                       'pad': self.pad,
                       'pad_type':self.pad_type,
                       'use_bias':self.use_bias,
                       'sn':self.sn})
        return config

class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True,
                 sn=False, name='FullyConnected'):
        super(FullyConnected, self).__init__(name=name)
        self.units = units
        self.use_bias = use_bias
        self.sn = sn

        if self.sn:
            self.fc = SpectralNormalization(tf.keras.layers.Dense(
                        self.units, kernel_initializer=weight_initializer,
                        kernel_regularizer=weight_regularizer_fully,
                        use_bias=self.use_bias), name='sn_' + self.name)
        else:
            self.fc = tf.keras.layers.Dense(self.units,
                        kernel_initializer=weight_initializer,
                        kernel_regularizer=weight_regularizer_fully,
                        use_bias=self.use_bias, name=self.name)

    def call(self, x, training=None, mask=None):
        x = tf.keras.layers.Flatten()(x)
        x = self.fc(x)
        return x

    def get_config(self):
        config = super(FullyConnected, self).get_config()
        config.update({"units": self.units,
                       'use_bias':self.use_bias,
                       'sn':self.sn})
        return config

class AdaIN(tf.keras.layers.Layer):
    def __init__(self, channels, sn=False, epsilon=1e-5, name='AdaIN'):
        super(AdaIN, self).__init__(name=name)
        self.channels = channels
        self.epsilon = epsilon
        self.gamma_fc = FullyConnected(self.channels,True, sn=sn)
        self.beta_fc = FullyConnected(self.channels, True, sn=sn)

    def call(self, x_init, training=True, mask=None):
        x, style = x_init
        x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_std = tf.sqrt(x_var + self.epsilon)
        x_norm = ((x - x_mean) / x_std)
        gamma = self.gamma_fc(style)
        beta = self.beta_fc(style)
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, self.channels])
        beta = tf.reshape(beta, shape=[-1, 1, 1, self.channels])
        x = (1 + gamma) * x_norm + beta
        return x

class SpectralNormalization(tf.keras.layers.Wrapper):
    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError('Please initialize `TimeDistributed` '
                             'layer with a `Layer` instance. You '
                             'passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape=None):
        self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                trainable=False, name=self.name + '_u', dtype=tf.float32,
                aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        super(SpectralNormalization, self).build()

    def call(self, inputs, training=None, mask=None):
        self.update_weights()
        output = self.layer(inputs)
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        u_hat = self.u
        v_hat = None
        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_ ** 2) ** 0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_ ** 2) ** 0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.layer.kernel.assign(self.w / sigma)

"""========================= Blocks ============================"""
class CrissCross(tf.keras.layers.Layer):
    def __init__(self,channels_in, channels_out, normalize=False,
                 downsample=False, use_bias=True, sn=False, name='ResBlock'):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.normalize = normalize
        self.downsample = downsample
        self.use_bias = use_bias
        self.sn = sn
        self.skip_flag = channels_in != channels_out
        self.query_conv = Conv(self.channels_in,)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, normalize=False,
                 downsample=False, use_bias=True, sn=False, name='ResBlock'):
        super(ResBlock, self).__init__(name=name)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.normalize = normalize
        self.downsample = downsample
        self.use_bias = use_bias
        self.sn = sn
        self.skip_flag = channels_in != channels_out

        self.conv_0 = Conv(self.channels_in, 3, 1, 1,
                           use_bias=self.use_bias,
                           sn=self.sn, name='conv_0')
        self.ins_norm_0 = InstanceNorm(name='instance_norm_0')

        self.conv_1 = Conv(self.channels_out, 3, 1, 1,
                           use_bias=self.use_bias,
                           sn=self.sn, name='conv_1')
        self.ins_norm_1 = InstanceNorm(name='instance_norm_1')

        if self.skip_flag:
            self.skip_conv = Conv(self.channels_out, 1, 1,
                                  use_bias=False, sn=self.sn,
                                  name = 'skip_conv')

    def shortcut(self, x):
        if self.skip_flag: x = self.skip_conv(x)
        if self.downsample:
            x = tf.keras.layers.AvgPool2D(2,2)(x)
        return x

    def residual(self, x):
        if self.normalize:x = self.ins_norm_0(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv_0(x)

        if self.downsample:
            x = tf.keras.layers.AvgPool2D((2,2))(x)
        if self.normalize: x = self.ins_norm_1(x)

        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv_1(x)
        return x

    def call(self, x_init, training=True, mask=None):
        x = self.residual(x_init) + self.shortcut(x_init)
        return x / tf.math.sqrt(2.0) # unit variance

    def get_config(self):
        config = super(ResBlock, self).get_config()
        config.update({"channels_in": self.channels_in,
                       'channels_out':self.channels_out,
                       'normalize': self.normalize,
                       'downsample': self.downsample,
                       'use_bias': self.use_bias,
                       'sn':self.sn})
        return config

class AdainResBlock(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, upsample=False,
                 use_bias=True, sn=False, name='AdainResBlock'):
        super(AdainResBlock, self).__init__(name=name)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.upsample = upsample
        self.use_bias = use_bias
        self.sn = sn

        self.skip_flag = channels_in != channels_out

        self.conv_0 = Conv(self.channels_out, 3, 1, 1,
                           use_bias=self.use_bias,
                           sn=self.sn, name='conv_0')
        self.adain_0 = AdaIN(self.channels_in, name='adain_0')
        self.conv_1 = Conv(self.channels_out, 3, 1, 1,
                           use_bias=self.use_bias,
                           sn=self.sn, name='conv_1')
        self.adain_1 = AdaIN(self.channels_out, name='adain_1')

        if self.skip_flag:
            self.skip_conv = Conv(self.channels_out, 1, 1,
                                  use_bias=False, sn=self.sn,
                                  name='skip_conv')

    def shortcut(self, x):
        if self.upsample:
            x = tf.keras.layers.UpSampling2D(size=(2, 2),
                                interpolation='bilinear')(x)
        if self.skip_flag: x = self.skip_conv(x)
        return x

    def residual(self, x, s):
        x = self.adain_0([x, s])
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        if self.upsample:
            x = tf.keras.layers.UpSampling2D(size=(2, 2),
                                interpolation='bilinear')(x)
        x = self.conv_0(x)

        x = self.adain_1([x, s])
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv_1(x)

        return x

    def call(self, x_init, training=True, mask=None):
        x_c, x_s = x_init
        x = self.residual(x_c, x_s) + self.shortcut(x_c)
        return x / tf.math.sqrt(2.0)

class UpResBlock(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out, normalize=False,
                 upsample=False, use_bias=True, sn=False, name='UpResBlock'):
        super(UpResBlock, self).__init__(name=name)
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.normalize = normalize
        self.upsample = upsample
        self.use_bias = use_bias
        self.sn = sn
        self.skip_flag = channels_in != channels_out

        self.conv_0 = Conv(self.channels_in, 3, 1, 1,
                           use_bias=self.use_bias,
                           sn=self.sn, name='conv_0')
        self.ins_norm_0 = InstanceNorm(name='instance_norm_0')

        self.conv_1 = Conv(self.channels_out, 3, 1, 1,
                           use_bias=self.use_bias,
                           sn=self.sn, name='conv_1')
        self.ins_norm_1 = InstanceNorm(name='instance_norm_1')

        if self.skip_flag:
            self.skip_conv = Conv(self.channels_out, 1, 1,
                                  use_bias=False, sn=self.sn,
                                  name = 'skip_conv')

    def shortcut(self, x):
        if self.skip_flag: x = self.skip_conv(x)
        if self.upsample:
            x = tf.keras.layers.UpSampling2D(size=(2, 2),
                            interpolation='bilinear')(x)
        return x

    def residual(self, x):
        if self.normalize:x = self.ins_norm_0(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv_0(x)

        if self.upsample:
            x = tf.keras.layers.UpSampling2D(size=(2, 2),
                            interpolation='bilinear')(x)
        if self.normalize: x = self.ins_norm_1(x)

        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.conv_1(x)
        return x

    def call(self, x_init, training=True, mask=None):
        x = self.residual(x_init) + self.shortcut(x_init)
        return x / tf.math.sqrt(2.0) # unit variance

    def get_config(self):
        config = super(UpResBlock, self).get_config()
        config.update({"channels_in": self.channels_in,
                       'channels_out':self.channels_out,
                       'normalize': self.normalize,
                       'upsample': self.upsample,
                       'use_bias': self.use_bias,
                       'sn':self.sn})
        return config

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, channels,
                 normalize=False,
                 sn=False,
                 name='ConvBlock'):
        super(ConvBlock, self).__init__(name=name)
        self.normalize = normalize

        self.ins_norm_0 = InstanceNorm('ins_norm_0')
        self.conv_0 = Conv(channels, 3, 1, 1,
                           use_bias=True,
                           sn=sn, name='conv_0')

        self.ins_norm_1 = InstanceNorm('ins_norm_1')
        self.conv_1 = Conv(channels, 3, 1, 1,
                           use_bias=True,
                           sn=sn, name='conv_1')

    def call(self, x, training=True, mask=None):
        y = self.ins_norm_0(x) if self.normalize else x
        y = tf.keras.layers.LeakyReLU(0.2)(y)
        y = self.conv_0(y)
        y = self.ins_norm_1(y) if self.normalize else y
        y = tf.keras.layers.LeakyReLU(0.2)(y)
        y = self.conv_1(y)

        return y

class DeconvBlock(tf.keras.layers.Layer):
    def __init__(self, channels,
                 normalize=False,
                 sn=False,
                 name='DeconvBlock'):
        super(DeconvBlock, self).__init__(name=name)
        self.normalize = normalize

        self.ins_norm_0 = InstanceNorm('ins_norm_0')
        self.conv_0 = Conv(channels, 3, 1, 1,
                           use_bias=True,
                           sn=sn, name='conv_0')

        self.ins_norm_1 = InstanceNorm('ins_norm_1')
        self.conv_1 = Conv(channels, 3, 1, 1,
                           use_bias=True,
                           sn=sn, name='conv_1')

    def call(self, x_init, training=True, mask=None):
        x, s = x_init
        y = self.ins_norm_0(x) if self.normalize else x
        y = tf.keras.layers.LeakyReLU(0.2)(y)
        y = tf.keras.layers.UpSampling2D(size=(2, 2),
                    interpolation='bilinear')(y)
        y = self.conv_0(y)

        y = tf.keras.layers.Concatenate(axis=-1)([y,s])

        y = self.ins_norm_1(y) if self.normalize else y
        y = tf.keras.layers.LeakyReLU(0.2)(y)
        y = self.conv_1(y)

        return y

class AddPriorBlock(tf.keras.layers.Layer):
    def __init__(self, channels_in, channels_out,
                 sn=False, name='AddPriorBlock'):
        super(AddPriorBlock, self).__init__(name=name)

        self.adain_norm = AdaIN(channels_in, name='adain_norm')
        self.conv_0 = Conv(channels_out, 3, 1, 1,
                           use_bias=True,
                           sn=sn, name='conv_0')

        self.ins_norm = InstanceNorm('ins_norm')
        self.conv_1 = Conv(channels_out, 3, 1, 1,
                           use_bias=True,
                           sn=sn, name='conv_1')

    def call(self, x_init, training=True, mask=None):
        x, s, p = x_init

        y = self.adain_norm([x, p])
        y = tf.keras.layers.LeakyReLU(0.2)(y)
        y = tf.keras.layers.UpSampling2D(size=(2, 2),
                        interpolation='bilinear')(y)
        y = self.conv_0(y)

        # Unet skip connection
        y = tf.keras.layers.Concatenate(axis=-1)([y, s])

        y = self.ins_norm(y)
        y = tf.keras.layers.LeakyReLU(0.2)(y)
        y = self.conv_1(y)

        return y
#-*- coding: UTF-8 -*-
"""
    Name: Zheng TANG
    Time: 2021/04/20
    Place: SIAT, Shenzhen
    Item:
"""

from numpy import log2
import tensorflow as tf
from ops import *
"""============================= 
        GAN baseline: ResNet 
==============================="""
class GAN_G(tf.keras.Model):
    def __init__(self, img_size=256,
                 img_ch=3, sn=False,
                 hidden_dims=512,
                 name='GAN_G'):
        #parent constructer
        super(GAN_G, self).__init__(name=name)

        #initialize layers
        self.hidden_dims, self.sn = hidden_dims, sn
        # For 256x256 image, init channel = 64
        # For 256x256 image, final maps = 16x16
        self.channels = 2 ** 14 // img_size
        self.repeat_num = int(log2(img_size)) - 4

        self.from_rgb = Conv(self.channels, 3, 1, 1,
                             sn=self.sn, name='from_rgb')
        self.to_rgb = tf.keras.Sequential([
            InstanceNorm(), tf.keras.layers.LeakyReLU(0.2),
            Conv(img_ch, 1, 1, sn=self.sn, name='to_rgb')])
        self.encoder, self.decoder = self.architecture_init()

    def architecture_init(self):
        ch_in = self.channels
        ch_out = self.channels
        encoder, decoder = [], []

        for i in range(self.repeat_num): # down/up-sampling blocks
            ch_out = min(ch_in * 2, self.hidden_dims)
            encoder.append(ResBlock(ch_in, ch_out, normalize=True,
                                    downsample=True, sn=self.sn,
                                    name='E_Resblock_' + str(i)))
            decoder.insert(0, UpResBlock(ch_out, ch_in, normalize=True,
                                        upsample=True, sn=self.sn,
                                        name='D_UpResBlock_'+ str(i)))
            ch_in = ch_out

        for i in range(2): # bottleneck blocks
            encoder.append(ResBlock(ch_out, ch_out,
                                    normalize=True, sn=self.sn,
                            name='E_bottleneck_resblock_' + str(i)))
            decoder.insert(0, UpResBlock(ch_out, ch_out,
                                         normalize=True, sn=self.sn,
                            name='D_bottleneck_upresblock_' + str(i)))

        return encoder, decoder

    def call(self, x_init, training=True, mask=None):
        x = self.from_rgb(x_init)
        for encoder_block in self.encoder:
            x = encoder_block(x)
        for decoder_block in self.decoder:
            x = decoder_block(x)
        x = self.to_rgb(x)
        return x

class GAN_D(tf.keras.Model):
    # PatchGAN, fully convolution network
    # many conv layer --> (B, 16,16,1)
    # Global Average pooling --> (B,1))
    def __init__(self,
                 img_size=256,
                 sn=False,
                 hidden_dims=512,
                 name='GAN_D'):
        super(GAN_D, self).__init__(name=name)
        self.hidden_dims, self.sn = hidden_dims, sn
        # if img_size=256, first channel=64
        # if img_size=256, final maps=16*16
        self.channels = 2 ** 14 // img_size
        self.repeat_num = int(log2(img_size)) - 4
        self.encoder = self.architecture_init()

    def architecture_init(self):
        ch_in = self.channels
        blocks = [Conv(ch_in, 3, 1, 1, sn=self.sn, name='init_conv')]

        for i in range(self.repeat_num):
            ch_out = min(ch_in * 2, self.hidden_dims)
            blocks += [ResBlock(ch_in, ch_out, downsample=True,
                                sn=self.sn, name='resblock_' + str(i))]
            ch_in = ch_out

        for i in range(2): # bottleneck blocks
            blocks += [ResBlock(ch_in, ch_in, normalize=True,
                                sn=self.sn, name='bottleneck_'+str(i))]

        blocks += [tf.keras.layers.LeakyReLU(alpha=0.2)]
        blocks += [Conv(1, 1, 1, sn=self.sn, name='conv_1')]
        blocks += [tf.keras.layers.GlobalAvgPool2D()] # [B,1,1,1]
        encoder = tf.keras.Sequential(blocks)
        return encoder

    def call(self, x_init, training=True, mask=None):
        x = self.encoder(x_init)
        return x

"""============================= 
        Unet-AutoEncoder 
==============================="""
class UNet_AE(tf.keras.Model):
    def __init__(self,
                 img_size=256,
                 img_ch=3,
                 sn=False,
                 hidden_dims=512,
                 name='UNet_AE'):
        super(UNet_AE, self).__init__(name=name)
        self.hidden_dims, self.sn = hidden_dims, sn
        self.channels = 2 ** 14 // img_size
        self.repeat_num = int(log2(img_size)) - 4

        self.from_rgb = Conv(self.channels, 3, 1, 1,
                             sn=sn, name='from_rgb')
        self.to_rgb = tf.keras.Sequential([
            InstanceNorm(), tf.keras.layers.LeakyReLU(0.2),
            Conv(img_ch, 1, 1, sn=self.sn, name='to_rgb')])
        self.encoder, self.bottleneck, self.decoder = self.networks()

    def networks(self):
        ch_in = self.channels
        encoder, bottle, decoder = [], [], []

        for i in range(self.repeat_num):
            encoder.append(ConvBlock(ch_in, normalize=True,
                                     sn=self.sn, name='E_%d'%i))
            decoder.insert(0, DeconvBlock(ch_in, normalize=True,
                                          sn=self.sn, name='D_%d'%i))
            ch_in = min(ch_in * 2, self.hidden_dims)

        for i in range(2): # bottleneck blocks
            bottle +=[ResBlock(ch_in, ch_in, normalize=True,
                                   sn=self.sn, name='bottleneck_%d'%i)]
        bottleneck = tf.keras.Sequential(bottle)

        return encoder, bottleneck, decoder

    def call(self, X, training=True, mask=None):
        x = self.from_rgb(X)
        copy = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            copy.append(x)
            x = tf.keras.layers.MaxPool2D((2,2))(x)
        x = self.bottleneck(x)
        copy.reverse()
        for i, decoder_block in enumerate(self.decoder):
            x = decoder_block([x, copy[i]])
        x = self.to_rgb(x)
        return x

"""============================= 
        Keras Models 
==============================="""
layers = tf.keras.layers

def Unet():
    input = tf.keras.Input(shape = (256, 256, 1), name='Input')
    # ---------------- Share MRI Encoder ----------------
    maps = layers.Lambda(lambda x: 2*x-1)(input)
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    s256 = maps
    maps = layers.MaxPool2D(name='s_128')(maps)

    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)
    s128 = maps
    maps = layers.MaxPool2D(name='s_64')(maps)

    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    s64 = maps
    maps = layers.MaxPool2D(name='s_32')(maps)

    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps) # 512--256
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps) # 512--256
    s32 = maps
    maps = layers.MaxPool2D(name='s_16')(maps)

    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)# 512--256
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)# 512--256
    s16 = maps
    maps = layers.MaxPool2D(name='s_8')(maps)

    # # ---------------- PET Decoder -----------------------
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)# 512--256
    maps = layers.UpSampling2D((2,2))(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)# 512--256
    maps = layers.Concatenate(name='s16_cat')([maps, s16])
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)# 512--256
    maps = layers.UpSampling2D((2,2))(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)# 512--256
    maps = layers.Concatenate(name='s32_cat')([maps, s32])
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)# 512--256

    maps = layers.UpSampling2D((2,2))(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Concatenate(name='s64_cat')([maps, s64])
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)

    maps = layers.UpSampling2D((2,2))(maps)
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Concatenate(name='s128_cat')([maps, s128])
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)

    maps = layers.UpSampling2D((2,2))(maps)
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Concatenate(name='s256_cat')([maps, s256])
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(1, 3, 1, 'same',activation='sigmoid')(maps)
    model = tf.keras.Model(inputs = input, outputs= maps, name = 'Unet')
    model.summary()
    return model

def Resnet50():
    input = tf.keras.Input(shape = (256, 256, 1), name='Input')
    X = layers.Concatenate()([input,input,input])
    X = layers.Lambda(lambda x: 2*x-1)(X)
    resnet = tf.keras.applications.ResNet50(include_top=False, weights=None,
                                            pooling=False, input_tensor=X)
    s08 = resnet.output
    # s16 = layers.Conv2DTranspose(512, 2, 2, 'same', activation='relu')(s08)
    # s32 = layers.Conv2DTranspose(256, 2, 2, 'same', activation='relu')(s16)
    # s64 = layers.Conv2DTranspose(128, 2, 2, 'same', activation='relu')(s32)
    # s128 = layers.Conv2DTranspose(64, 2, 2, 'same', activation='relu')(s64)
    # s256 = layers.Conv2DTranspose( 1, 2, 2, 'same', activation='sigmoid')(s128)

    s16 = layers.UpSampling2D((2,2))(s08)
    s16 = layers.Conv2D(512, 3,1,'same',activation='relu')(s16)
    s32 = layers.UpSampling2D((2,2))(s16)
    s32 = layers.Conv2D(256, 3, 1, 'same', activation='relu')(s32)
    s64 = layers.UpSampling2D((2,2))(s32)
    s64 = layers.Conv2D(128, 3, 1, 'same', activation='relu')(s64)
    s128 = layers.UpSampling2D((2,2))(s64)
    s128 = layers.Conv2D(64, 3, 1, 'same', activation='relu')(s128)
    s256 = layers.UpSampling2D((2,2))(s128)
    s256 = layers.Conv2D(1, 3, 1, 'same', activation='sigmoid')(s256)
    model = tf.keras.Model(inputs=input, outputs=s256, name='Resnet50')
    # model.summary()
    return model

def Residual_Unet():
    input = tf.keras.Input(shape=(256, 256, 1), name='Input')
    # ---------------- Share NAC Encoder ----------------
    maps = layers.Lambda(lambda x: 2 * x - 1)(input)
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    s256 = maps
    maps = layers.MaxPool2D(name='s_128')(maps)

    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)
    s128 = maps
    maps = layers.MaxPool2D(name='s_64')(maps)

    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    s64 = maps
    maps = layers.MaxPool2D(name='s_32')(maps)

    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)  # 512--256
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)  # 512--256
    s32 = maps
    maps = layers.MaxPool2D(name='s_16')(maps)

    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)  # 512--256
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)  # 512--256
    s16 = maps
    maps = layers.MaxPool2D(name='s_8')(maps)
    s08 = maps
    # ---------------- Residual Transform -----------------------
    for i in range(9):
        maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(s08)
        maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
        maps = layers.Add(name='residual_%d'%i)([maps, s08])
        s08 = maps

    # ---------------- Decoder -----------------------
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(s08)  # 512--256

    maps = layers.UpSampling2D((2, 2))(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)  # 512--256
    maps = layers.Concatenate(name='s16_cat')([maps, s16])
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)  # 512--256

    maps = layers.UpSampling2D((2, 2))(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)  # 512--256
    maps = layers.Concatenate(name='s32_cat')([maps, s32])
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)  # 512--256

    maps = layers.UpSampling2D((2, 2))(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Concatenate(name='s64_cat')([maps, s64])
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)

    maps = layers.UpSampling2D((2, 2))(maps)
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Concatenate(name='s128_cat')([maps, s128])
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)

    maps = layers.UpSampling2D((2, 2))(maps)
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Concatenate(name='s256_cat')([maps, s256])
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)

    maps = layers.Conv2D(1, 3, 1, 'same', activation='sigmoid')(maps)
    model = tf.keras.Model(inputs=input, outputs=maps, name='Residual_Unet')
    model.summary()
    return model

def VGG16_Unet():
    input = tf.keras.Input(shape=(256, 256, 1), name='Input')
    maps = layers.Lambda(lambda x: 2*tf.concat([x,x,x],-1)-1)(input)
    # 256 x 256 x 64
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu', name='block1_conv1')(maps)
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu', name='block1_conv2')(maps)
    s256 = maps
    # 128 x 128 x 128
    maps = layers.MaxPool2D(name='s_128')(maps)
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu', name='block2_conv1')(maps)
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu', name='block2_conv2')(maps)
    s128 = maps
    # 64 x 64 x 256
    maps = layers.MaxPool2D(name='s_64')(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv1')(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv2')(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu', name='block3_conv3')(maps)
    s064 = maps
    # 32 x 32 x 512
    maps = layers.MaxPool2D(name='s_32')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv1')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv2')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block4_conv3')(maps)
    s032 = maps
    # 16 x 16 x 512
    maps = layers.MaxPool2D(name='s_16')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv1')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv2')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu', name='block5_conv3')(maps)

    s016 = maps
    for i in range(3):
        maps = layers.Conv2D(64, 1, 1, 'same', activation='relu')(s016)
        maps = layers.Conv2D(512, 3, 1, 'same',activation='relu')(maps)
        maps = layers.Add(name='residual_%d'%i)([maps, s016])
        s016 = maps
    # maps = layers.Conv2D(512, 3, 1,'same',activation='relu')(maps)
    # maps = layers.Concatenate(name='global_maps')([maps, s016])
    # maps = layers.Conv2D(512, 3, 1, 'same', activation='relu')(maps)

    # 16 x 16 x 512
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu')(maps)
    # 32 x 32 x 512
    maps = layers.UpSampling2D((2,2),name='d_32')(maps)
    maps = layers.Concatenate(name='cat_32')([maps, s032])
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(512, 3, 1, 'same', activation='relu')(maps)
    # 64 x 64 x 256
    maps = layers.UpSampling2D((2,2),name='d_64')(maps)
    maps = layers.Concatenate(name='cat_64')([maps, s064])
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(256, 3, 1, 'same', activation='relu')(maps)
    # 128 x 128 x 128
    maps = layers.UpSampling2D((2,2),name='d_128')(maps)
    maps = layers.Concatenate(name='cat_128')([maps, s128])
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(128, 3, 1, 'same', activation='relu')(maps)
    # 256 x 256 x 64
    maps = layers.UpSampling2D((2,2),name='d_256')(maps)
    maps = layers.Concatenate(name='cat_256')([maps, s256])
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    maps = layers.Conv2D(64, 3, 1, 'same', activation='relu')(maps)
    # 256 x 256 x 1
    maps = layers.Conv2D(1, 3, 1, 'same', activation='sigmoid')(maps)
    model = tf.keras.Model(inputs=input, outputs=maps, name='VGG_Unet')

    model.load_weights('./weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    for layer in model.layers[2:19]:
        layer.trainable = False
    model.summary()

    return model

def D_Pix2Pix(with_target=True):
    input = tf.keras.Input(shape = (256, 256, 1))
    if with_target:
        label = tf.keras.Input(shape = (256, 256, 1))
        d0 = layers.Concatenate(axis=-1)([label, input])
    else:
        d0 = input
    d0 = layers.Lambda(lambda x: 2*x-1)(d0)
    d1 = layers.Conv2D(64, 4, 2, 'same')(d0)
    d1 = layers.LeakyReLU()(d1)
    d2 = layers.Conv2D(64*2, 4, 2, 'same')(d1)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.LeakyReLU()(d2)
    d3 = layers.Conv2D(64*4, 4, 2, 'same')(d2)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.LeakyReLU()(d3)
    d4 = layers.Conv2D(64*4, 4, 2, 'same')(d3)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.LeakyReLU()(d4) # [16,16]
    validity = layers.Conv2D(1, 4, 1, 'same')(d4)
    if with_target:
        model = tf.keras.Model([input,label], validity,
                               name='Discriminator')
    else:
        model = tf.keras.Model(input, validity,
                               name='Discriminator')
    # model.summary()
    return model

def D_WGAN():
    input = tf.keras.Input(shape = (256, 256, 1))
    maps = layers.Conv2D(64, 3, 2, 'same')(input)
    maps = layers.BatchNormalization(momentum=0.8)(maps)
    maps = layers.LeakyReLU(0.2)(maps) #[128]
    maps = layers.Conv2D(128, 3, 2, 'same')(maps)
    maps = layers.BatchNormalization(momentum=0.8)(maps)
    maps = layers.LeakyReLU(0.2)(maps) #[64]
    maps = layers.Conv2D(128, 3, 2, 'same')(maps)
    maps = layers.BatchNormalization(momentum=0.8)(maps)
    maps = layers.LeakyReLU(0.2)(maps) #[32]
    maps = layers.Conv2D(256, 3, 2, 'same')(maps)
    maps = layers.BatchNormalization(momentum=0.8)(maps)
    maps = layers.LeakyReLU(0.2)(maps) #[16]
    maps = layers.Conv2D(256, 3, 2, 'same')(maps)
    maps = layers.BatchNormalization(momentum=0.8)(maps)
    maps = layers.LeakyReLU(0.2)(maps) #[8]
    maps = layers.Conv2D(512, 3, 2, 'same')(maps)
    maps = layers.BatchNormalization(momentum=0.8)(maps)
    maps = layers.LeakyReLU(0.2)(maps) #[4]
    maps = layers.Flatten()(maps)
    # maps = layers.Dense(1024)(maps)
    # maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Dense(1)(maps)
    model = tf.keras.Model(input, maps, name='D_WGAN')
    # model.summary()
    return model

def D_WGANGP():
    input = tf.keras.Input(shape = (256, 256, 1))
    maps = layers.Lambda(lambda x: 2*x-1)(input)
    maps = layers.Conv2D(32, 3, 1, 'same')(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Conv2D(32, 3, 2, 'same')(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Conv2D(64, 3, 1, 'same')(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Conv2D(64, 3, 2, 'same')(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Conv2D(128, 3, 1, 'same')(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Conv2D(128, 3, 2, 'same')(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Conv2D(256, 3, 1, 'same')(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Conv2D(256, 3, 2, 'same')(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Flatten()(maps)
    maps = layers.Dense(1024)(maps)
    maps = layers.LeakyReLU(0.2)(maps)
    maps = layers.Dense(1)(maps)
    model = tf.keras.Model(input, maps, name='D_WGANGP')
    # model.summary()
    return model

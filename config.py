#-*- coding: UTF-8 -*-
"""
    Name: Zheng Tang
    Time: 2021/04/27
    Place: SIAT, Shenzhen
    Item: MR ----> PET

"""
import tensorflow._api.v2.compat.v1 as tfv

tfv.app.flags.DEFINE_string(
    'GPU', '0', 'the order of using GPU')

tfv.app.flags.DEFINE_string(
    'logdir', './logs/baseline/unet_MR2CT_2021-0426-1059',
    'the dir for tensorboard log files')

tfv.app.flags.DEFINE_string(
    'train_dir', './data/PT_MR_CENTRAL2/axial',
    'the file of train dataset')
tfv.app.flags.DEFINE_string(
    'test_dir', './data/2021-06-13/axial',
    'the file of test dataset')
tfv.app.flags.DEFINE_string(
    'valid_dir', './data/2021-06-13/axial',
    'the file of test dataset')
tfv.app.flags.DEFINE_string(
    'data_dir', './data/',
    'the dir of train/valid/test dataset')

tfv.app.flags.DEFINE_string(
    'input_name', 'MRI',
    'the name of input name, one of ["NAC","umap"]')

tfv.app.flags.DEFINE_string(
    'output_name', 'PET',
    'the name of output name, one of ["AC","CT"]')

tfv.app.flags.DEFINE_integer(
    'batch_size', 2, 'The training batch size.')

tfv.app.flags.DEFINE_integer(
    'img_size', 256, 'The size of images.')

tfv.app.flags.DEFINE_integer(
    'img_ch', 1, 'The channels of images.')

tfv.app.flags.DEFINE_integer(
    'num_epoch', 600, 'The number of epoach in train.')

tfv.app.flags.DEFINE_float(
    'gen_lr', 1e-4, 'Initial learning rate in generator.')

tfv.app.flags.DEFINE_float(
    'dis_lr', 1e-4, 'Initial learning rate in discriminator.')

tfv.app.flags.DEFINE_bool(
    'finetune', True, 'whether fine tuning or not')

tfv.app.flags.DEFINE_string(
    'phase', 'test', 'whether train phase or test phase')

tfv.app.flags.DEFINE_integer(
    'memory_limit', None, 'whether train phase or test phase')

FLAGS = tfv.app.flags.FLAGS

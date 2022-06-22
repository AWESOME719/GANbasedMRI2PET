# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
"""
    Name: Zheng TANG
    Time: 2021/04/10
    Place: SIAT, Shenzhen
    Item: MRI --> PET

"""

import tensorflow as tf
from config import FLAGS

"""==================================================
                        Configure
=================================================="""
import os, shutil


def run_on_gpu():
    # tf.enable_v2_behavior()
    # os.environ["CUDA_VISIBLE_DEVICES"] = '/device:GPU:0'
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
    # print(FLAGS.GPU)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Number of GPU Hardwares in YOUR COMPUTER
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    # assert len(GPUs) > 0, "Not enough GPU hardware devices available"
    print(GPUs)
    # Limit GPU memory
    if FLAGS.memory_limit and FLAGS.memory_limit > 1:  # unit: MB
        tf.config.experimental.set_virtual_device_configuration(GPUs[0],
                                                                [tf.config.experimental.VirtualDeviceConfiguration(
                                                                    memory_limit=FLAGS.memory_limit)])
    else:
        tf.config.experimental.set_memory_growth(GPUs[0], True)

    # Number of (virtual) GPU cores applied in running codes
    # avaiable_gpu_cores = tf.config.experimental.list_logical_devices('GPU')
    # Logging the run allocation info in terminate
    # tf.debugging.set_log_device_placement(True)


def check_logdir():
    if FLAGS.phase == 'train':
        if os.path.exists(FLAGS.logdir):
            if not FLAGS.finetune:
                shutil.rmtree(FLAGS.logdir)
                print(FLAGS.logdir + ' has been removed')
    else:
        if not os.path.exists(FLAGS.logdir):
            raise NameError(FLAGS.logdir + ' does not exist')


"""==================================================
                        DataSet Preprocessing
=================================================="""


def adjust_windows(img, WL=300, WW=1500,
                   intercept=-1024):
    # Chest window: [-160, 240] HU
    # Bone window: [-450, 1050] HU
    im = (img + intercept)
    UL = WL + WW / 2.0
    DL = WL - WW / 2.0
    UL_mask = im > UL
    DL_mask = im <= DL
    img_new = (im - DL) / WW
    img_new[UL_mask] = 1
    img_new[DL_mask] = 0
    return img_new


def squarized(img):
    """
    crop customized data into square
    """
    H, W = img.shape[0], img.shape[1]
    LP = (W - H) // 2
    im = img[0:H,
         LP:LP + H]
    return im


def crop_square_by_axis(img, crop_y_location=(20, 256)):
    """
    crop image with specifized location in square shape
    """
    if img.ndim == 2:
        H, W = img.shape[0], img.shape[1]
        TP_up = crop_y_location[0]
        TP_bottom = crop_y_location[1]
        length = (TP_bottom - TP_up) // 2
        x_mid = W // 2
        im = img[TP_up: TP_bottom, x_mid - length:x_mid + length]
        return im
    if img.ndim == 3:
        H, W = img.shape[0], img.shape[1]
        TP_up = crop_y_location[0]
        TP_bottom = crop_y_location[1]
        length = (TP_bottom - TP_up) // 2
        # y_MID = (TP_bottom - TP_up) // 2
        x_mid = W // 2
        im = img[TP_up: TP_bottom, x_mid - length:x_mid + length]
        return im


def central_crop(img, crop_size=(256, 256)):
    """
    crop from middle with square-shaped size of crop_size
    """
    if img.ndim == 2:
        H, W = img.shape[0], img.shape[1]
        LP = (W - crop_size[1]) // 2
        TP = (H - crop_size[0]) // 2
        im = img[TP:TP + crop_size[0],
             LP:LP + crop_size[1]]
        return im

    if img.ndim == 3:  # [H, W, C]
        H, W = img.shape[0], img.shape[1]
        LP = (W - crop_size[1]) // 2
        TP = (H - crop_size[0]) // 2
        im = img[TP:TP + crop_size[0],
             LP:LP + crop_size[1], :]
        return im

    if img.ndim == 4:  # [B,H,W,C]
        H, W = img.shape[1], img.shape[2]
        LP = (W - crop_size[1]) // 2
        TP = (H - crop_size[0]) // 2
        im = img[:, TP:TP + crop_size[0],
             LP:LP + crop_size[1], :]
        return im


"""==================================================
                        DataSet Loader
=================================================="""
from glob import glob
import random
import pandas as pd


def glob_image_from_folder(folder, shuffle=False):
    image_list = glob(folder + '/*png')
    if shuffle:
        image_list = random.shuffle(image_list)
    return image_list


def split_dataset(paths):
    random.shuffle(paths)
    num_train = 8 * len(paths) // 10
    num_valid = len(paths) // 20
    train = paths[:num_train]
    test = paths[num_train: -num_valid]
    valid = paths[-num_valid:]
    return train, valid, test


def read_pairs_with_augment(path1, path2):
    input = tf.io.read_file(path1)
    label = tf.io.read_file(path2)
    input = tf.image.decode_png(input, dtype=tf.dtypes.uint16)
    label = tf.image.decode_png(label, dtype=tf.dtypes.uint16)
    # input = tf.image.convert_image_dtype(input, tf.float32)
    # label = tf.image.convert_image_dtype(label, tf.float32)

    # Normalization
    input = 2 * (input / 1500) - 1
    label = 2 * (label / 12000) - 1

    # surrounding Pad
    paddings = tf.constant([[15, 15], [15, 15], [0, 0]])
    input = tf.pad(input, paddings, "SYMMETRIC")
    label = tf.pad(label, paddings, "SYMMETRIC")

    # Random Crop
    stack = tf.stack([input, label], axis=0)
    crop = tf.image.random_crop(stack, [2,
                                        FLAGS.img_size,
                                        FLAGS.img_size,
                                        FLAGS.img_ch])
    input, label = crop[0], crop[1]

    # Conditional horizontal Flip
    if tf.random.uniform(()) > 0.5:
        input = tf.image.flip_left_right(input)
        label = tf.image.flip_left_right(label)

    return (input, label)


def read_pairs_without_augment(path1, path2):
    input = tf.io.read_file(path1)
    label = tf.io.read_file(path2)
    input = tf.image.decode_jpeg(input, FLAGS.img_ch)
    label = tf.image.decode_jpeg(label, FLAGS.img_ch)
    input = tf.image.convert_image_dtype(input, tf.float32)
    label = tf.image.convert_image_dtype(label, tf.float32)

    # Normalization
    input = 2 * input - 1
    label = 2 * label - 1
    return (input, label)


def read_image_with_augment(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, FLAGS.img_ch, dtype=tf.dtypes.uint16)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    # input, label = tf.split(image, 2, axis=1)
    label, input = tf.split(image, 2, axis=1)

    # Normalization
    input = 2 * (input / 1500) - 1  # MRI
    # label = 2 * (label/10000) - 1 #PET
    label = 2 * (label / 12000) - 1  # CT

    # surrounding Pad
    paddings = tf.constant([[15, 15], [15, 15], [0, 0]])
    input = tf.pad(input, paddings, "SYMMETRIC")
    label = tf.pad(label, paddings, "SYMMETRIC")

    # Random Crop
    stack = tf.stack([input, label], axis=0)
    crop = tf.image.random_crop(stack, [2,
                                        FLAGS.img_size,
                                        FLAGS.img_size, FLAGS.img_ch
                                        ])
    input, label = crop[0], crop[1]

    # Conditional horizontal Flip
    if tf.random.uniform(()) > 0.5:
        input = tf.image.flip_left_right(input)
        label = tf.image.flip_left_right(label)

    return (input, label)


def read_image_without_augment(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, FLAGS.img_ch, dtype=tf.dtypes.uint16)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    # input, label = tf.split(image, 2, axis=1) #[H, W, C]
    label, input = tf.split(image, 2, axis=1)  # [H, W, C]
    input = 2 * (input / 1500) - 1
    label = 2 * (label / 12000) - 1
    return (input, label)


def dataset_loader(data, map_fun):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(len(data),
                              reshuffle_each_iteration=True).repeat()
    dataset = dataset.map(map_fun, AUTOTUNE)
    dataset = dataset.batch(FLAGS.batch_size,
                            drop_remainder=True)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def keras_generator(X, Y, train=True):
    # X, Y are Numpy or H5 data
    if train:
        Augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
            zoom_range=0.2, data_format="channels_last",
            rotation_range=30, fill_mode='reflect',
            width_shift_range=0.15, height_shift_range=0.15,
            horizontal_flip=True, vertical_flip=True, )
    else:
        Augmentor = tf.keras.preprocessing.image.ImageDataGenerator(
            data_format="channels_last")
    seed = random.randint(0, 1000)
    genX = Augmentor.flow(X, batch_size=FLAGS.batch_size, seed=seed)
    genY = Augmentor.flow(Y, batch_size=FLAGS.batch_size, seed=seed)
    while True: yield genX.next(), genY.next()


"""==================================================
                        Model Tools
=================================================="""


def LR_schedule(iterations, init_lr, type='Piecewise'):
    if type == 'Piecewise':
        phase = [int(i * iterations) for i in [0.5, 0.6, 0.7, 0.8, 0.9]]
        decay_lr = [i * init_lr for i in [1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            phase, decay_lr)
    elif type == 'Cycle':
        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            init_lr, iterations * 0.4, 1e-6, power=0.5, cycle=True)
    elif type == 'Direct':
        lr_schedule = init_lr

    else:
        raise ValueError(type + ' is not pre-defined.')
    return lr_schedule


def save_csv(file_name, contents):
    # contents: dict {}, like:
    # df = pd.DataFrame({'list1': list1,
    #                    'list2': list2})
    if type(contents).__name__ == 'dict':
        df = pd.DataFrame(contents)
    else:
        raise ValueError('The function receives only DICT input')

    if '.csv' in file_name:
        df.to_csv(file_name)
    else:
        raise ValueError('The saved file must be CSV file')


def fprint(file, info):
    print(info)
    with open(file, 'a') as the_file:
        the_file.write(info + '\n')


def list_mean_std(data, multiple=1):
    n = len(data)
    mean = sum(data) / n
    variance = sum([((x - mean) ** 2) for x in data]) / n
    stddev = variance ** 0.5
    return mean * multiple, stddev * multiple


"""==================================================
                        Loss and Metrics
=================================================="""


def MSM(y_true, y_pred):
    # Adjust brightness and contrast
    true = scale_to_minmax(y_true, 0, 1)
    pred = scale_to_minmax(y_pred, 0, 1)
    ms_ssim = tf.image.ssim_multiscale(true,
                                       pred,
                                       1.0)
    return 1 - tf.reduce_mean(ms_ssim)


def BME(y_true, y_pred):
    BCE = tf.keras.losses.BinaryCrossentropy()
    mask = tf.cast(y_true > 0, tf.float32)
    bce = BCE(y_true=mask, y_pred=y_pred)
    mae = tf.reduce_sum(tf.abs(y_true - y_pred * mask)) / tf.reduce_sum(mask)
    return bce + 100 * mae


class PCP():
    def __init__(self, model, layers):
        model.trainable = False
        self.layers = layers
        outputs = [model.get_layer(L).output for L in self.layers]
        self.extractor = tf.keras.Model([model.input], outputs)
        self.loss_object = tf.keras.losses.MeanSquaredError()

    def Loss(self, y_true, y_pred):
        if y_true.shape[-1] == 1:
            true = tf.concat([y_true, y_true, y_true], -1)
            pred = tf.concat([y_pred, y_pred, y_pred], -1)
        else:
            true = y_true
            pred = y_pred

        content_pred = self.extractor(pred)
        content_true = self.extractor(true)
        content_loss = 0.0
        for i in range(len(self.layers)):
            content_loss += self.loss_object(content_true[i],
                                             content_pred[i])
        content_loss /= len(self.layers)
        return content_loss


def NRMSE(x, y, keep_dims=False):
    nmse = tf.reduce_mean(tf.square(y - x), [1, 2, 3])
    rmse = tf.sqrt(nmse)
    MAX = tf.reduce_max(y, [1, 2, 3])
    MIN = tf.reduce_min(y, [1, 2, 3])
    nrmse = tf.math.divide_no_nan(rmse, MAX - MIN)
    if keep_dims:
        return nrmse
    else:
        return tf.reduce_mean(nrmse)


# def RMSE(x,y):
#     nmse = tf.reduce_mean(tf.square(y-x))
#     rmse = tf.sqrt(nmse)
#     return rmse

# def NMAE(x,y,Mean=True):
#     axis = [0, 1, 2, 3] if Mean else [1, 2, 3]
#     mae = tf.reduce_mean(tf.abs(y-x),axis=axis)
#     range = tf.reduce_max(y, axis) - tf.reduce_min(y, axis)
#     nmae = tf.math.divide_no_nan(mae, range)
#     return nmae
def PSNR(x, y, keep_dims=False):
    tensor_x = scale_to_minmax(x)
    tensor_y = scale_to_minmax(y)
    psnr = tf.image.psnr(tensor_x, tensor_y, 1.0)
    if keep_dims:
        return psnr
    else:
        return tf.reduce_mean(psnr)


def SSIM(x, y, keep_dims=False):
    tensor_x = scale_to_minmax(x)
    tensor_y = scale_to_minmax(y)
    ssim = tf.image.ssim(tensor_x, tensor_y, 1.0)
    if keep_dims:
        return ssim
    else:
        return tf.reduce_mean(ssim)


def PCC(x, y, keep_dims=False):
    if len(x.shape) == 4:
        xm = x - tf.reduce_mean(x, [1, 2, 3], True)
        ym = y - tf.reduce_mean(y, [1, 2, 3], True)
        r_num = tf.reduce_sum(xm * ym, [1, 2, 3])
        x_square_sum = tf.reduce_sum(xm * xm, [1, 2, 3])
        y_square_sum = tf.reduce_sum(ym * ym, [1, 2, 3])
    else:
        xm = x - tf.reduce_mean(x)
        ym = y - tf.reduce_mean(y)
        r_num = tf.reduce_sum(xm * ym)
        x_square_sum = tf.reduce_sum(xm * xm)
        y_square_sum = tf.reduce_sum(ym * ym)
    r_den = tf.sqrt(x_square_sum * y_square_sum)
    r = tf.math.divide_no_nan(r_num, r_den)
    if keep_dims:
        return r
    else:
        return tf.reduce_mean(r)


"""==================================================
                        Keras Tools
=================================================="""


def scale_to_minmax(img, min=0, max=1):
    if len(img.shape) == 4:
        MIN = tf.reduce_min(img, [1, 2, 3], keepdims=True)
        MAX = tf.reduce_max(img, [1, 2, 3], keepdims=True)
    else:
        MIN = tf.reduce_min(img)
        MAX = tf.reduce_max(img)
    scale = tf.math.divide_no_nan(img - MIN, MAX - MIN)
    return (max - min) * scale + min


class Tensorboard(tf.keras.callbacks.Callback):
    def __init__(self, dataset, log_dir, loss_names, metric_names, shuffle=True):
        super().__init__()
        self.input, self.target = dataset
        self.loss_names = loss_names
        self.metric_names = metric_names
        self.shuffle, self.num_summary = shuffle, 4
        self.train_summary_writer = tf.summary.create_file_writer(log_dir + '/train')
        self.valid_summary_writer = tf.summary.create_file_writer(log_dir + '/valid')

    def on_epoch_end(self, epoch, logs=None):
        if self.shuffle:
            concat = tf.concat((self.input, self.target), 2)
            shuffle = tf.random.shuffle(concat)
            inputs, targets = tf.split(shuffle, 2, 2)
            input = inputs[:self.num_summary]
            target = targets[:self.num_summary]
        else:
            input = self.input[:self.num_summary]
            target = self.target[:self.num_summary]

        logit = self.model.predict_on_batch(input)
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)

        with self.train_summary_writer.as_default():
            tf.summary.scalar('Learning_rate', lr, epoch)
            tf.summary.scalar('loss/total', logs['loss'], epoch)

            if self.loss_names:
                for loss in self.loss_names:
                    tf.summary.scalar('loss/' + loss, logs[loss], epoch)

            if self.metric_names:
                for metric in self.metric_names:
                    tf.summary.scalar('metric/' + metric, logs[metric], epoch)

        with self.valid_summary_writer.as_default():
            tf.summary.scalar('loss/total', logs['val_loss'], epoch)

            if self.loss_names:
                for loss in self.loss_names:
                    tf.summary.scalar('loss/' + loss, logs['val_' + loss], epoch)

            if self.metric_names:
                for metric in self.metric_names:
                    tf.summary.scalar('metric/' + metric, logs['val_' + metric], epoch)

            tf.summary.image('Input', input, epoch, self.num_summary)
            tf.summary.image('Target', target, epoch, self.num_summary)
            tf.summary.image('Logit', logit, epoch, self.num_summary)


"""==================================================
                        DataSet Post-Processing
=================================================="""
from patchify import patchify, unpatchify


def tf_split_to_patches(image, patch_size, patch_step):
    # image must be 2D or 2D numpy array
    # [B,H,W,C] --> [nb,nh,nw,nc, b,h,w,c]
    if isinstance(patch_step, int):
        patch_x_step = patch_step
        patch_y_step = patch_step
    else:
        patch_x_step = patch_step[0]
        patch_y_step = patch_step[1]
    if (image.shape[0] - patch_size[0]) % patch_x_step != 0:
        raise ValueError('The required condition for perfect patchify'
                         ' is to have (width - patch_width) mod step_size = 0.')
    elif (image.shape[1] - patch_size[1]) % patch_y_step != 0:
        raise ValueError('The required condition for perfect patchify'
                         ' is to have (height - patch_height) mod step_size = 0.')
    else:
        patches = patchify(image.numpy(), patch_size, step=patch_step)
    return patches


def tf_merge_from_patches(patches, merge_shape):
    # image must be 2D or 3D numpy array
    # [nb,nh,nw,nc, b,h,w,c] --> [B,H,W,C]
    image = unpatchify(patches.numpy(), merge_shape)
    return image


import PIL.Image


def draw_PIL_canvas(save_path, image_list, hist,
                    normalize=False):
    # image_list:[ [I1], [I2  [I4
    #                     I3]  I5
    #                          I6]   }
    # hist: [1,2, 3]
    num_cols = len(hist)
    num_rows = max(hist)
    I = image_list[0][0]
    width = num_cols * I.shape[1]
    height = num_rows * I.shape[0]
    canvas = PIL.Image.new('RGB', (width, height),
                           color=(153, 153, 153))
    for col in range(num_cols):
        num_row = hist[col]
        for row in range(num_row):
            pos_y = col * I.shape[1]
            pos_x = row * I.shape[0]
            image = image_list[col][row]
            image = tf_minmax_norm(image, 0, 255) \
                if normalize else image * 255
            im = PIL.Image.fromarray(image.numpy(), 'RGB')
            canvas.paste(im, (pos_y, pos_x))
    canvas.save(save_path)


def draw_tf_canvas(save_path, image_list, row, col):
    images = tf.stack(image_list, 0)
    # images must be [B, H, W, C] shape
    # batch_size >= size * size
    if len(images) < row * col:
        raise ValueError('images are not enough to build canvas')
    else:
        t = tf.unstack(images[:row * col], num=row * col, axis=0)
        rows = [tf.concat(t[i * row:(i + 1) * row], 0) for i in range(col)]
        canvas = tf.concat(rows, axis=1)  # [s*H, s*W, C]
        tf.keras.preprocessing.image.save_img(save_path, canva)
    return canvas


import pandas as pd


def save_csv(file_name, contents):
    # contents: dict {}, like:
    # df = pd.DataFrame({'list1': list1,
    #                    'list2': list2})
    if type(contents).__name__ == 'dict':
        df = pd.DataFrame(contents)
    else:
        raise ValueError('The function receives only DICT input')

    if '.csv' in file_name:
        df.to_csv(file_name)
    else:
        raise ValueError('The saved file must be CSV file')


def list_mean_std(data, multiple=1):
    n = len(data)
    mean = sum(data) / n
    variance = sum([((x - mean) ** 2) for x in data]) / n
    stddev = variance ** 0.5
    return mean * multiple, stddev * multiple

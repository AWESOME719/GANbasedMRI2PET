from scipy.io import savemat
from datetime import datetime
from copy import deepcopy
import tensorflow as tf
import cv2
import numpy as np
from config import FLAGS
from utils import *
from models import *
import matplotlib.pyplot as plt

class MR_CT():
    def __init__(self):
        self.path = 'F:/PET-MR-TRANS/outs/baseline/unet_MR2PT_centre1/0022.png'
        self.upperpath = 'F:/PET-MR-TRANS/outs/baseline/unet_MR2PT_centre1'
    def SplitImage(self):

        image = tf.io.read_file(self.path)
        image = tf.image.decode_png(image, FLAGS.img_ch, dtype=tf.dtypes.uint16)
        p0, p1, p2 = tf.split(image, 3, axis=1)
        p0 = p0.numpy()
        p1 = p1.numpy()
        p2 = p2.numpy()
        save_name_0 = self.upperpath + '/OriginalInput'
        if not os.path.exists(save_name_0):
            os.makedirs(save_name_0)
        save_name_1 = self.upperpath + '/label'
        if not os.path.exists(save_name_1):
            os.makedirs(save_name_1)
        save_name_2 = self.upperpath + '/logit'
        if not os.path.exists(save_name_2):
            os.makedirs(save_name_2)
        print(save_name_0)
        cv2.imwrite(save_name_0 + '/' + '4.png', p0.astype(np.uint16))
        cv2.imwrite(save_name_1 + '/' + '4.png', p1.astype(np.uint16))
        cv2.imwrite(save_name_2 + '/' + '4.png', p2.astype(np.uint16))

if __name__ == '__main__':
    m = MR_CT()
    m.SplitImage()

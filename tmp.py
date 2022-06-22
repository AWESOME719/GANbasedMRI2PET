#-*- coding: UTF-8 -*-
"""
    Item: Medical Image Synthesis
"""
import numpy as np, cv2
import shutil, os
from utils import *
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import numpy as np

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
# import colormaps as cmaps

class MyPlot():
    def __init__(self):
        self.title = 'default'
        # SetUP
        self.fontprop = font_manager.FontProperties(
            family='Times New Roman',
            style='normal', weight='bold')
        self.font = 'Times New Roman'
        params = {'legend.fontsize': 22,
                  'figure.figsize': (12, 5),
                  'axes.labelsize': 22,  # 轴名称
                  'axes.titlesize': 18,
                  'xtick.labelsize': 24,  # 轴数值
                  'ytick.labelsize': 26,
                  'figure.titlesize': 18,
                  }
        plt.rcParams.update(params)
        plt.style.use('seaborn-whitegrid')
        sns.set_style("white")

        """==================== Prepare data ======================="""
        # p0 = 'F:/PET-MR-TRANS/outs/baseline/unet_MR2PT_centre1/0009.png'
        # # p0_img = tf.io.read_file(p0)
        # p0_mid = tf.io.read_file(p0)
        # p0_img = tf.image.decode_png(p0_mid, 1, dtype=tf.dtypes.uint16)
        # p0_1, p0_2, p0_3 = tf.split(p0_img, 3, axis=1)
        # p0_1_np = p0_1.numpy()
        # p0_2_np = p0_2.numpy()
        # p0_3_np = p0_3.numpy()
        # p1 = p0_2_np.astype(np.uint16)
        # p2 = p0_3_np.astype(np.uint16)`
        # self.x1 = p1
        # self.x2 = p2
        p1 = 'F:\\PET-MR-TRANS\\outs\\baseline\\unet_MR2PT_centre1\\\label\\4.png'
        p2 = 'F:\\PET-MR-TRANS\\outs\\baseline\\unet_MR2PT_centre1\\\logit\\4.png'
        # # p1 = './outs/01051500-RWCNNv0/tmp/sagittal/ct/005_0010.png'
        # # p2 = './outs/01051500-RWCNNv0/tmp/sagittal/sct/005_0010.png'
        # # self.x0 = cv2.imread(p0, cv2.IMREAD_UNCHANGED)
        self.x1 = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
        self.x2 = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
        # self.x1 = p2.numpy()
        # self.x2 = p3.numpy()
        # self.x3 = p1.numpy() #Original Input
        # self.L0 = self.x0[387, 27: 237] * 1.1e-3
        # self.L1 = self.x1[387, 27: 237] * 1.1e-3
        # self.L2 = self.x2[387, 27: 237] * 1.1e-3
        # self.r1, self.r2 = 27, 237



    def Datacase(self, modality = 'ac', case='007'):
        prefix_path = '.outs/baseline/unet_MR2CT_2021-0426-1059'
        path_true = glob('F:\\PET-MR-TRANS\\outs\\baseline\\unet_MR2PT_202104301414\\label' + '\\*.png')
        # print(path_true)
        path_fake = glob('F:\\PET-MR-TRANS\\outs\\baseline\\unet_MR2PT_202104301414\\logit' + '\\*.png')
        true, fake = [], []
        for p1 in path_true:
            # print(p1)
            image = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
            true.append(image)
        for p2 in path_fake:
            image = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
            fake.append(image)
        return np.reshape(true,[-1,]), np.reshape(fake, [-1,])

    def NAC_AC_PET(self):
        prefix_path = './outs/01051500-RWCNNv0/tmp/axial/'
        path_NAC = glob(prefix_path + 'nac/005_*.png')
        path_AC = glob(prefix_path + 'ac/005_*.png')
        NAC, AC = [], []
        for p1 in path_NAC[:90]:
            image = cv2.imread(p1, cv2.IMREAD_UNCHANGED)
            NAC.append(image)
        for p2 in path_AC[:90]:
            image = cv2.imread(p2, cv2.IMREAD_UNCHANGED)
            AC.append(image)
        return np.reshape(NAC, [-1, ]), np.reshape(AC, [-1, ])

    def Histogram(self, title=None, xlabel=None, ylabel=None):
        """
        title: str
        xlabel: label of xtick
        ylabel: label of ytick
        """
        # Prepare data
        # x0 = np.reshape(self.x0, [-1,])
        title = self.title
        x1 = np.reshape(self.x1, [-1,])
        x2 = np.reshape(self.x2, [-1,])
        # x0, x1 = self.NAC_AC_PET()
        # x1, x2 = self.Datacase('ac', '007')
        print(np.max(x1), np.max(x2))

        x1 = x1 * (1.5/2000)
        x2 = x2 * (1.9/2000)
        seq = np.arange(0, 2.01, 0.5)

        plt.figure(figsize=(10, 9),dpi=100)
        # figsize=(10,9), fontsize=40,42
        # figsize=(12,5), fontsize=32,34
        _, _, _ = plt.hist(x1, 1000, density=True,
                           facecolor='g', alpha=0.2)
        _, _, _ = plt.hist(x2, 1000, density=True,
                           facecolor='r',alpha=0.2)
        # _, _, _ = plt.hist(x0, 200, density=True,
        #                    facecolor='b', alpha=0.2)
        # plt.xticks(seq, fontproperties = self.font, size=30)
        plt.xticks(fontproperties = self.font, size=30)
        plt.yticks(fontproperties = self.font, size=30)
        if title: plt.title(title)
        if xlabel: plt.xlabel(xlabel)
        if ylabel: plt.ylabel(ylabel)
        plt.yscale('log')
        # plt.grid(True)
        plt.show()

    def DensityPlot(self, title=None):
        # title = self.title
        plt.figure(dpi=80)
        # color = ['g', 'deeppink', 'dodgerblue', 'orange']
        sns.kdeplot(self.x1, shade=True, color="dodgerblue", alpha=.7)
        sns.kdeplot(self.x2, shade=True, color="deeppink", alpha=.7)
        if title: plt.title(title)
        plt.xlim(200, 2000)
        plt.show()

    def DensityHist(self, title=None):
        title = self.title
        plt.figure(dpi=80)
        sns.distplot(self.x1, color="dodgerblue",
                     hist_kws={'alpha': .7},
                     kde_kws={'linewidth': 3})
        sns.distplot(self.x2, color="orange",
                     hist_kws={'alpha': .7},
                     kde_kws={'linewidth': 3})
        if title: plt.title(title)
        plt.title('Whole Body DensityHist')
        plt.xlabel('Density')
        plt.show()

    def Profile(self, interval = 50):
        x = np.arange(self.r1, self.r2, 1)
        xx = np.arange(self.r1, self.r2, interval)
        # yy = np.arange(0, 4000, 500)

        plt.figure(figsize=(8, 5),dpi=100)
        plt.plot(x, self.L0, linewidth=1.5, linestyle='-.', color='blue')
        plt.plot(x, self.L1, linewidth=1.5, linestyle='-', color='g')
        plt.plot(x, self.L2, linewidth=1.5, linestyle=':', color='m')
        plt.fill_between(x, self.L1, self.L2, where=self.L2 >= self.L1,
                         facecolor='purple', interpolate=True, alpha=0.35)
        plt.fill_between(x, self.L1, self.L2, where=self.L2 <= self.L1,
                        facecolor='g', interpolate=True, alpha=0.35)
        plt.xticks(xx, fontproperties = self.font, size=22)
        plt.yticks(fontproperties = self.font, size=22)
        plt.xlim(self.r1, self.r2)
        plt.grid(True)
        plt.show()

    def ErrorMap(self):
        image1 = self.x1.astype(np.float)
        image2 = self.x2.astype(np.float)
        # plt.imshow(image1)
        # plt.imshow(image2)
        # plt.show()
        # image2 = cv2.resize(image2, (262,262))
        # image1 = cv2.resize(image1, (262,262))
        error = image2 - image1
        # error = np.transpose(image2 - image1)
        print(np.min(error), np.max(error))
        plt.figure(figsize=(15,15),dpi=100)
        plt.imshow(error,
                   vmin=0, vmax=+1000,
                   cmap=plt.get_cmap('bwr'))
        plt.axis('off')
        plt.colorbar(orientation='vertical')
        plt.show()

    def JointHist(self):
        # 005 case for CT, start=0, end=573
        # 007 case for AC, start=1174, end=1742
        x1, x2 = self.Datacase()
        print(len(x1), len(x2))
        xx1, xx2 = zip(*random.sample(
            list(zip(x1 * 9.9e-4,
                     x2 * 9.9e-4)), 50000))
        seq = np.arange(0, 75, 10)
        plt.figure(figsize=(10,9), dpi=100)
        plt.scatter(xx2, xx1, marker='o',  s=10,
                    c='cornflowerblue', alpha=0.5)
        plt.plot(np.arange(0, 70, 1),
                 np.arange(0, 70, 1),
                 linewidth=4, linestyle='--',
                 color='red', alpha = 0.9)
        plt.xticks(seq, fontproperties=self.font, size=15)
        plt.yticks(seq, fontproperties=self.font, size=15)
        plt.xlabel('Synthetic PET')
        plt.ylabel('REAL PET')
        plt.grid(True)
        plt.show()




if __name__ == '__main__':
    # tmp()
    # tmp2()
    # tmp3()
    # tmp4()
    # tmp5()
    # tmp6()
    # tmp7()
    # tmp8()
    # tmp9()
    # tmp10()
    # tmp11()
    # tmp12()
    # tmp13()
    # tmp14()
    # tmp15()
    # MyPlot().Histogram()
    # MyPlot().Profile()
    # MyPlot().DensityHist()
    # MyPlot().JointHist()
    MyPlot().ErrorMap()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
# import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pylab as mpl
import numpy as np
import matplotlib.dates as mdates
class MyPlot():
    def __init__(self):
        self.fontprop = font_manager.FontProperties(
            family='Times New Roman',
            style='normal', weight='bold')
        self.font = 'Times New Roman'
        params = {'legend.fontsize': 15,
                  'figure.figsize': (12, 5),
                  'axes.labelsize': 15,  # 轴名称
                  'axes.titlesize': 18,
                  'xtick.labelsize': 15,  # 轴数值
                  'ytick.labelsize': 15,
                  'figure.titlesize': 18,
                  'lines.linewidth': 0.5
                  }
        plt.rcParams.update(params)
        plt.style.use('seaborn-whitegrid')
        # sns.set_style("white")
        """==================== Prepare data ======================="""
        self.s1_p = [57.41518021, 59.27973175, 55.3416214, 60.39251709, 57.56972504, 55.10539627,57.79912949, 59.35772324, 55.42263031, 61.29601288, 60.9563446]
        self.s1_u = [41.98384476, 48.09879684, 47.51156616, 43.08202362, 45.79478836, 43.16246414, 48.69735336,41.83594131, 44.39845276, 49.28234863, 41.81274414]
        self.s1_g = [31.36543274, 43.42895126, 31.3325119, 41.28331375, 30.46450806, 39.36017227, 32.88866806,34.65792084, 34.33415985, 36.4510231, 39.49942017]
        self.s2_p = [48.50024033, 53.74185562, 53.53123856, 47.22397614, 55.30761719, 48.31663132,51.09715652, 46.97610474, 57.8393631, 55.04837799, 54.42288208]
        self.s2_u = [45.42288208,42.14271545,47.31335831,40.46308517,47.99838257,40.8732872,37.77912903,39.65615845,41.05200195,40.17185593,40.27162552]
        self.s2_g = [37.36543274, 37.42895126, 36.3325119, 36.28331375, 35.46450806, 34.36017227, 33.88866806,33.65792084, 35.33415985, 37.4510231, 38.49942017]
        self.s1_p_ssim = [0.998291492,0.998343587,0.998238087,0.99758333,0.997372568,0.995887339,0.989922106,0.983760118,0.978200197,0.973830283,0.970394313]
        self.s1_u_ssim = [0.989970386,0.98691982,0.985252619,0.981908798,0.974547088,0.963197351,0.954262376,0.944398403,0.927353799,0.941252053,0.993120253]
        self.s1_g_ssim = [0.989922106,0.983760118,0.978200197,0.973830283,0.970394313,0.969203651,0.969596386,0.97113806,0.971936226,0.975850284,0.987763574]
        self.s2_p_ssim = [0.992505312,0.990055919,0.986826301,0.984401226,0.982486784,0.9775635,0.974318087,0.970531821,0.968304992,0.970160186,0.97544688]
        self.s2_u_ssim = [0.981908798,0.974547088,0.963197351,0.954262376,0.944398403,0.927353799,0.941252053,0.996901393,0.997215986,0.997273207,0.997530103]
        self.s2_g_ssim = [0.987482131,0.984625578,0.980549037,0.978595257,0.974009693,0.964972138,0.954891562,0.941814065,0.932700634,0.925161839,0.940439165]
        self.s1_p_mae = [26.21711731,28.01906586,27.80702209,26.11141205,23.95101929,26.31721497,28.27212524,24.26100159,21.60675049,27.86863708,22.03614807]
        self.s1_u_mae = [31.30046082, 35.47349548
                         , 40.48170471
                         , 38.76542664
                         , 39.10999298
                         , 33.35894775
                         , 44.97077942
                         , 48.18367004
                         , 43.96603394
                         , 39.57367706
                         , 37.39234924]
        self.s1_g_mae = [60.08990479
            , 70.01931763
            , 82.67041016
            , 90.43661499
            , 96.68759155
            , 90.1913757
            , 76.89181519
            , 75.21694946
            , 80.03134155
            , 73.1582489
            , 70.91294861
                         ]
        self.s2_p_mae = [24.2657547
            ,28.30265045
            ,31.30046082
            ,35.47349548
            ,40.48170471
            ,38.76542664
            ,39.10999298
            ,33.35894775
            ,20.7924118
            ,44.97077942
            ,48.18367004
            ]
        self.s2_u_mae = [39.57367706
            ,37.39234924
            ,31.62572479
            ,32.64572144
            ,31.79423523
            ,36.64047241
            ,43.54751587
            ,50.70957947
            ,52.96002197
            ,48.751297
            ,40.1698761
            ]
        self.s2_g_mae =[52.04601288
            ,43.57626343
            ,48.32766724
            ,48.90814209
            ,41.81994629
            ,43.25822449
            ,42.73999786
            ,42.0188446
            ,44.05754852
            ,50.38855743
            ,45.80247498
            ]



    def box_chart(self):
        box_1 = self.s1_p
        box_2 = self.s1_u
        box_3 = self.s1_g
        box_4 = self.s2_p
        box_5 = self.s2_u
        box_6 = self.s2_g
        box_7 = self.s1_p_ssim
        box_8 = self.s1_u_ssim
        box_9 = self.s1_g_ssim
        box_10 = self.s2_p_ssim
        box_11 = self.s2_u_ssim
        box_12 = self.s2_g_ssim
        box_13 = self.s1_p_mae
        box_14 = self.s1_u_mae
        box_15 = self.s1_g_mae
        box_16 = self.s2_p_mae
        box_17 = self.s2_u_mae
        box_18 = self.s2_g_mae
        # df = pd.DataFrame(data)
        fig = plt.figure()
        x = fig.add_subplot(111)
        site_1 = ['Site 1' for _ in range(11)]
        site_2 = ['Site 2' for _ in range(11)]
        model_p = ['Proposed Model' for _ in range(11)]
        model_u = ['U-Net' for _ in range(11)]
        model_g = ['GAN' for _ in range(11)]
        s1_p_psnr = pd.DataFrame({'PSNR': box_1, 'site': site_1, 'Model': model_p})
        s1_u_psnr = pd.DataFrame({'PSNR': box_2, 'site': site_1, 'Model': model_u})
        s1_g_psnr = pd.DataFrame({'PSNR': box_3, 'site': site_1, 'Model': model_g})
        s2_p_psnr = pd.DataFrame({'PSNR': box_4, 'site': site_2, 'Model': model_p})
        s2_u_psnr = pd.DataFrame({'PSNR': box_5, 'site': site_2, 'Model': model_u})
        s2_g_psnr = pd.DataFrame({'PSNR': box_6, 'site': site_2, 'Model': model_g})
        s1_p_ssim = pd.DataFrame({'SSIM': box_7, 'site': site_1, 'Model': model_p})
        s1_u_ssim = pd.DataFrame({'SSIM': box_8, 'site': site_1, 'Model': model_u})
        s1_g_ssim = pd.DataFrame({'SSIM': box_9, 'site': site_1, 'Model': model_g})
        s2_p_ssim = pd.DataFrame({'SSIM': box_10, 'site': site_2, 'Model': model_p})
        s2_u_ssim = pd.DataFrame({'SSIM': box_11, 'site': site_2, 'Model': model_u})
        s2_g_ssim = pd.DataFrame({'SSIM': box_12, 'site': site_2, 'Model': model_g})
        s1_p_mae = pd.DataFrame({'MAE': box_13, 'site': site_1, 'Model': model_p})
        s1_u_mae = pd.DataFrame({'MAE': box_14, 'site': site_1, 'Model': model_u})
        s1_g_mae = pd.DataFrame({'MAE': box_15, 'site': site_1, 'Model': model_g})
        s2_p_mae = pd.DataFrame({'MAE': box_16, 'site': site_2, 'Model': model_p})
        s2_u_mae = pd.DataFrame({'MAE': box_17, 'site': site_2, 'Model': model_u})
        s2_g_mae = pd.DataFrame({'MAE': box_18, 'site': site_2, 'Model': model_g})
        data1 = pd.concat([s1_p_psnr, s1_u_psnr, s1_g_psnr, s2_p_psnr, s2_u_psnr, s2_g_psnr])
        data2 = pd.concat([s1_p_ssim, s1_u_ssim, s1_g_ssim, s2_p_ssim, s2_u_ssim, s2_g_ssim])
        data3 = pd.concat([s1_p_mae, s1_u_mae, s1_g_mae, s2_p_mae, s2_u_mae, s2_g_mae])
        # print(data.head())
        # x = sns.boxplot(x='site', y='PSNR', data=data, hue='Model', width=0.5, linewidth=1.0, palette='Set3')
        # x = sns.boxplot(x='site', y='SSIM', data=data2, hue='Model', width=0.5, linewidth=1.0, palette='Set3')
        x = sns.boxplot(x='site', y='MAE', data=data3, hue='Model', width=0.5, linewidth=1.0, palette='Set3')
        # x.grid()
        x.legend().set_visible(False)
        # sns.show()
        # plt.figure(figsize=(10, 5))  # 设置画布的尺寸
        # fig = plt.figure()
        # x = fig.add_subplot(111)
        # # x.title('Examples of boxplot', fontsize=20)  # 标题，并设定字号大小
        # #
        # # x.boxplot([box_1,box_4], patch_artist=True, boxprops={'color': 'orangered', 'facecolor': 'pink'})
        # # x.set(xticklabels=['site 1','site 2'])
        # # x.set(title='Proposed Model')
        # x.boxplot([box_3,box_6], patch_artist=True, boxprops={'color': 'orangered', 'facecolor': 'pink'})
        # x.set(xticklabels=['site 1','site 2'])
        # x.set(title='GAN')
        #  # x2 = x.twinx()
        # # x2.boxplot([box_4, box_5, box_6], patch_artist=True, boxprops={'color': 'orangered', 'facecolor': 'blue'})
        plt.show()  # 显示图像

if __name__ == '__main__':
    MyPlot().box_chart()

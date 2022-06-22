import matplotlib.font_manager as font_manager
# import seaborn as sns
import warnings; warnings.filterwarnings(action='once')
import matplotlib.pyplot as plt
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
                  'xtick.labelsize': 10,  # 轴数值
                  'ytick.labelsize': 10,
                  'figure.titlesize': 18,
                  'lines.linewidth': 0.5
                  }
        plt.rcParams.update(params)
        plt.style.use('seaborn-whitegrid')
        # sns.set_style("white")
        """==================== Prepare data ======================="""
        # p0 = './outs/01051500-RWCNNv0/tmp/coronal/nac/007_0007.png'
        self.y1 = [0.0565749, 0.045275632, 0.029786188, 0.056022574, 0.071946412, 0.061456498, 0.073483586, 0.08917587, 0.069235539, 0.05115546, 0.038335635, 0.04723023]
        self.y2 = [0.079338512, 0.060532518, 0.042604107, 0.084076238, 0.09817723, 0.103964758, 0.115204903, 0.109409268, 0.099358859, 0.075778327, 0.071002292, 0.08122336]
        self.y3 = [0.098094386, 0.075962865, 0.058244784, 0.102468043, 0.118319492, 0.125663802, 0.135510023, 0.133744757, 0.11705999, 0.096151401, 0.091910413, 0.106950453]
        self.y4 = [0.129721946, 0.093290661, 0.076878352, 0.123343771, 0.140141073, 0.146870075, 0.158019596, 0.162046036, 0.142712885, 0.121835599, 0.113633069, 0.132617515]
        self.y5 = [0.176701904, 0.139503129, 0.116395298, 0.163533885, 0.181019419, 0.186507006, 0.198548022, 0.202516235, 0.187653263, 0.176833079, 0.151414614, 0.178515501]
        self.y6 = [0.0412426, 0.0520642, 0.1818066, 0.2286873, 0.1251359, -0.063988, -0.141308, -0.1239125, -0.0529252, 0.1728544, 0.0378742, 0.0488971]
        self.x1 =['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
        # self.xs = [datetime.strptime(d, '%Y').date() for d in self.x1]
        self.xs = self.x1
    def line_chart(self):

        y1 = self.y1
        y2 = self.y2
        y3 = self.y3
        y4 = self.y4
        y5 = self.y5
        y6 = self.y6
        xs = self.xs
        plt.figure(figsize=(10, 9), dpi=100)

        fig = plt.figure()
        x = fig.add_subplot(111)
        x.plot(xs, y1, 'k>--', label='IVOL 1')
        x.plot(xs, y2, 'k*--', label='IVOL 2')
        x.plot(xs, y3, 'k+--', label='IVOL 3')
        x.plot(xs, y4, 'kx--', label='IVOL 4')
        x.plot(xs, y5, 'kp--', label='IVOL 5')
        x.legend(bbox_to_anchor=(0.03, -0.15 , 1., .5), loc='lower left',
           ncol=6, borderaxespad=0.)
        x.grid()
        x2 = x.twinx()
        x2.plot(xs, y6, 'k-',linewidth=1,label='R')
        plt.rcParams['font.sans-serif'] = ['STSong']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        mpl.rcParams['font.size'] = 8
        x.set_xticks(xs)
        x2.legend(bbox_to_anchor=(0.9, -0.15 , 1. , .5), loc='lower left',
           ncol=1, mode="expand", borderaxespad=0.)
        x.set_ylim(0,0.25)
        x2.set_ylabel('超额收益率')
        seq = np.arange(0, 0.25, 0.05)
        seq1 = np.arange(-0.1, 0.25, 0.25)
        x2.set_ylim(-0.15,0.25)
        # x.yticks(seq, size=10)
        # x2.xticks(xs, size=15)
        # x2.yticks(seq1, size=10)
        # plt.xticks(, fontproperties=self.font, size=40)
        # plt.axis('off')
        # plt.title('面板数据')
        # x.set_xlabel('时间')
        x.set_ylabel('组合加权特质波动率')
        # ystring= u'aaaaa'
        # tx.text(0, 0.5, '\n'.join(ystring.replace('-', '')), va='center')
        # plt.legend(loc = 0, prop = {'size':10})

        plt.show()


if __name__ == '__main__':
    MyPlot().line_chart()
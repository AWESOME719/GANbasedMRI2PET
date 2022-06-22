#-*- coding: UTF-8 -*-
"""
    Name: Zheng TANG
    Time: 2021/02/23
    Place: SIAT, Shenzhen
    Item: Medical Image Synthesis
"""
"""
Registration require assistant of Radiant DICOM reader
"""
import numpy as np, cv2, os, random, shutil
from scipy.ndimage import zoom
from pydicom import read_file
from matplotlib import pyplot as plt
from utils import adjust_windows
from utils import central_crop
# from utils import hist_norm
from PIL import Image
from utils import squarized
from utils import crop_square_by_axis
# class TB_PET_CT_data():
#     def __init__(self):
#         self.prefix_path = 'I:/PET-CT Cases/'
#         self.NAC_CT_dir = './data/NAC_CT/'
#         self.AC_CT_dir = './data/AC_CT/'
#         self.selects = [list(range(5-1,651)),list(range(3-1,649)),
#                       list(range(6-1,652)),list(range(7-1,653)),
#                       list(range(23-1,669)),list(range(5-1,651)),
#                       list(range(4-1,650)),list(range(5-1,651)),
#                       list(range(6-1,652)),list(range(5-1,651)),
#                       list(range(4-1,650)),list(range(4-1,650)),
#                       list(range(7-1,653)),list(range(7-1,653)),
#                       list(range(7-1,653)),list(range(9-1,655)),
#                       list(range(7-1,653)),list(range(8-1,654))]
#         self.keeps = [[55-1,618],[13-1,567],[78-1,631],[88-1,590],
#                       [50-1,622],[33-1,633],[55-1,622],[103-1,609],
#                       [71-1,610],[66-1,626],[78-1,617],[100-1,590],
#                       [104-1,590],[91-1,614],[89-1,603],[89-1,590],
#                       [72-1,630], [74-1,620],[82-1,634],[53-1,633]]
#
#     def array_of_case(self, dir, select=None):
#         """
#
#         :param dir: directory of dicom data
#         :param select:
#         :return:
#         """
#         files = os.listdir(dir)
#         arrays = []
#         for f in files:
#             p = os.path.join(dir, f)
#             I = read_file(p).pixel_array
#             arrays.append(I)
#         array = np.array(arrays)
#         if select: array = array[select]
#         return np.transpose(array, [1, 2, 0])
#
#     def save_3d_array(self, folder ='001'):
#         PNAC = self.prefix_path + folder + '/NAC'
#         PAC = self.prefix_path + folder + '/AC'
#         PCT = self.prefix_path + folder + '/CT'
#         select = self.selects[int(folder)-3]
#         NAC = self.array_of_case(PNAC)
#         AC = self.array_of_case(PAC)
#         CT = self.array_of_case(PCT, select)
#         print('Finish loading', NAC.shape, AC.shape, CT.shape)
#         NAC =zoom(NAC, (434/256, 434/256, 1))
#         AC = zoom(AC, (434/256, 434/256, 647/673))
#         CT = zoom(CT, (362/512, 362/512, 1))
#         print('Finish resizing', NAC.shape, AC.shape, CT.shape)
#         NAC = central_crop(NAC, (256,256))
#         AC = central_crop(AC, (256,256))
#         CT = central_crop(CT, (256,256))
#         print('Finish cropping', NAC.shape, AC.shape, CT.shape)
#         np.savez(self.prefix_path + folder + '_data.npz',
#                  NAC = NAC, AC = AC, CT = CT)
#         print('Finish saving', folder, np.max(NAC), np.max(AC))
#
#     def generate_2d_images(self,file):
#         # load data
#         data = np.load(self.prefix_path + file)
#         prefix_name, _ = file.split('_')
#         NAC, AC, CT = data['NAC'], data['AC'], data['CT']
#
#         # normalization
#         # CT = 255.0 * adjust_window(CT, 300, 1500) # CT bone
#
#         # Axial slices
#         A1_dir = self.NAC_CT_dir + 'axial'
#         A2_dir = self.AC_CT_dir + 'axial'
#         if not os.path.exists(A1_dir): os.makedirs(A1_dir)
#         if not os.path.exists(A2_dir): os.makedirs(A2_dir)
#
#         start, end = self.keeps[int(prefix_name) - 1]
#         for i in range(start, end):
#             print('Axial', prefix_name, i)
#             # if i < start + 40:
#             #     nac = 255.0 * (1 - NAC[:,:,i] / np.max(NAC[:,:,i]))
#             #     ac = 255.0 * (1 - AC[:, :, i] / np.max(AC[:, :, i]))
#             # else:
#             #     nac = np.clip(NAC[:,:,i], 0, 2000)
#             #     ac = np.clip(AC[:,:,i], 0, 5000)
#             #     nac = 255.0 * (1 - nac / np.max(nac))
#             #     ac = 255.0 * (1 - ac / np.max(ac))
#             # ac = 255.0 * hist_norm(AC[:,:,i], reverse=True)
#             # comb = np.concatenate((ac, CT[:,:,i]), 1)
#             # cv2.imwrite(axial_dir + '/' + prefix_name +
#             #             '_%04d.jpg'%(i-start), comb)
#             A1 = np.concatenate((NAC[:, :, i], CT[:, :, i]),1)
#             cv2.imwrite(A1_dir + '/' + prefix_name + '_%04d.png'
#                         % (i - start), A1.astype(np.uint16))
#             if prefix_name not in ['001', '002', '003', '004']:
#                 A2 = np.concatenate((AC[:, :, i], CT[:, :, i]), 1)
#                 cv2.imwrite(A2_dir + '/' + prefix_name + '_%04d.png'
#                             % (i - start), A2.astype(np.uint16))
#
#         # normalization
#         # NAC = np.clip(NAC, 0, 2000)
#         # AC = np.clip(AC, 0, 5000)
#
#         # Coronal slices
#         C1_dir = self.NAC_CT_dir + 'coronal'
#         C2_dir = self.AC_CT_dir + 'coronal'
#         if not os.path.exists(C1_dir): os.makedirs(C1_dir)
#         if not os.path.exists(C2_dir): os.makedirs(C2_dir)
#
#         for i in range(155, 186):
#             print('Coronal', prefix_name, i)
#             nac = cv2.resize(NAC[i,:,:], (256 * 5, 256))[:, 30:30+832]
#             ct = cv2.resize(CT[i, :, :], (256 * 5, 256))[:, 30:30+832]
#             ac = cv2.resize(AC[i, :, :], (256 * 5, 256))[:, 30:30+832]
#             # nac = 255.0 * (1 - nac/np.max(nac))
#             # ac = 255.0 * (1 - ac/np.max(ac))
#             # ac = 255.0 * hist_norm(ac, True)
#             # comb = np.concatenate((np.transpose(ac), np.transpose(ct)), 1)
#             # cv2.imwrite(coronal_dir + '/' + prefix_name + '_%04d.jpg'%(i-155), comb)
#             C1 = np.concatenate((np.transpose(nac), np.transpose(ct)),1)
#             cv2.imwrite(C1_dir + '/' + prefix_name + '_%04d.png'
#                         % (i - start), C1.astype(np.uint16))
#             if prefix_name not in ['001', '002', '003', '004']:
#                 C2 = np.concatenate((np.transpose(ac), np.transpose(ct)), 1)
#                 cv2.imwrite(C2_dir + '/' + prefix_name + '_%04d.png'
#                             % (i - start), C2.astype(np.uint16))
#
#         # Sagittal slices
#         S1_dir = self.NAC_CT_dir + 'sagittal'
#         S2_dir = self.AC_CT_dir + 'sagittal'
#         if not os.path.exists(S1_dir): os.makedirs(S1_dir)
#         if not os.path.exists(S2_dir): os.makedirs(S2_dir)
#
#         for i in range(115, 146):
#             print('Sagittal', prefix_name, i)
#             nac = cv2.resize(NAC[:,i,:], (256 * 5, 256))[:, 30:30+832]
#             ct = cv2.resize(CT[:, i, :], (256 * 5, 256))[:, 30:30+832]
#             ac = cv2.resize(AC[:, i, :], (256 * 5, 256))[:, 30:30+832]
#             # nac = 255.0 * (1 - nac/np.max(nac))
#             # ac = 255.0 * (1 - ac/np.max(ac))
#             # ac = 255.0 * hist_norm(ac, True)
#             # comb = np.concatenate((np.transpose(ac), np.transpose(ct)), 1)
#             # cv2.imwrite(sagittal_dir + '/' + prefix_name + '_%04d.jpg'%(i-115), comb)
#             S1 = np.concatenate((np.transpose(nac), np.transpose(ct)),1)
#             cv2.imwrite(S1_dir + '/' + prefix_name + '_%04d.png'
#                         % (i - start), S1.astype(np.uint16))
#             if prefix_name not in ['001', '002', '003', '004']:
#                 S2 = np.concatenate((np.transpose(ac), np.transpose(ct)), 1)
#                 cv2.imwrite(S2_dir + '/' + prefix_name + '_%04d.png'
#                             % (i - start), S2.astype(np.uint16))
#                 """
#                 Registration between PET and CT
#                 """


class WB_PET_MR_data():
    """
    Registration between PET and MR data sequnce
    """
    def __init__(self):
        self.prefix_path = 'F:/PET-MR-TRANS/RAW_DATA_OLD' #原数据存储地址
        self.save_dir = './data/MR_PET_RZYQ/' #处理后数据保存地址

    def array_of_case(self, dir, select=None, reverse=False):
        """

        :param dir: directory of source data
        :param select: truncation of source data
        :param reverse:negative film
        :return:data series after proccessing
        """
        files = os.listdir(dir)
        if reverse: files.reverse()
        arrays = []
        for f in files:
            p = os.path.join(dir, f)
            I = read_file(p).pixel_array
            arrays.append(I)
        array = np.array(arrays)
        if select: array = array[select]
        return np.transpose(array, [1, 2, 0]) # transpose input array into regular order

    def save_3d_array(self, folder='001'):
        """
        :param folder: subfolder in the source data directory, it depends on the situation in each cases
        :return:image after registrarion stored as '.npz' temporary
        This function first resize two sequence into same size( both x,y,z )
         then crop them from the central point.
        """
        PET = self.prefix_path + '/' + folder + '/CT'
        PMR = self.prefix_path + '/' + folder + '/MR'
        # PT  = self.array_of_case(PET, reverse=True)
        PT  = self.array_of_case(PET, list(range(2-1,110)))
        PT = central_crop(PT,(113,113))

        MR = self.array_of_case(PMR, list(range(2-1,156)))#range wiped out unneccesarry part e.g. top of skull

        # MR = self.array_of_case(PMR, reverse=True)
        MR = central_crop(MR, (348, 348))
        MR = squarized(MR)
        #range wiped out unneccesarry part e.g. top of skull

        print('Finish loading', PT.shape, MR.shape)
        # PT = zoom(PT, (328/256, 328/256, 164/227)) #first
        PT = zoom(PT, (1, 1, 1))#first
        # three parameter each corresponding to zooming ratio on each direction
        # depends on different case, the ratio can be learnt from radiant automatic zooming operation
        MR = zoom(MR, (113/381, 113/381, 109/155))

        print('Finish resizing', PT.shape, MR.shape)
        # PT = central_crop(PT, (261,261))
        # MR = central_crop(MR, (306,306))

        print('Finish cropping', PT.shape, MR.shape)
        np.savez(self.prefix_path + folder + '_data.npz',
                 PT = PT, MR = MR)

        print('Finish saving', folder + '_data.npz')

    def generate_2d_images(self, file, resize=False):
        # load data
        data = np.load(self.prefix_path + file)
        prefix_name, _ = file.split('_')
        PT, MR = data['PT'], data['MR']
        PT = np.clip(PT, 0, 12000)
        # normalization
        # MR = 255.0 * np.clip((MR / 250.0), 0, 1)

        # Axial slices
        axial_dir = self.save_dir + 'axial'
        if not os.path.exists(axial_dir):
            os.makedirs(axial_dir)
        for i in range(1, 109):
            print('Axial', prefix_name, i)
            if resize:
                ac = cv2.resize(PT[:,:,i], (256,256))
                mr = cv2.resize(MR[:,:,i], (256,256))
            else:
                ac = PT[:,:,i]
                mr = MR[:,:,i]
            # if i > 85: ac = np.clip(ac, 0, 5000)
            # ac = 255.0 * (1 - ac / np.max(ac))
            comb = np.concatenate((ac, mr), 1)
            # comb = -comb
            cv2.imwrite(axial_dir + '/' + prefix_name + '_%04d.png'
                        %i, comb.astype(np.uint16))


        # normalization
        # AC = np.clip(AC, 0, 12000)

        # Coronal slices
        # coronal_dir = self.save_dir + 'coronal'
        # if not os.path.exists(coronal_dir): os.makedirs(coronal_dir)
        # for i in range(1, 100):
        #     print('Coronal', prefix_name, i)
        #     if resize:
        #         ac = cv2.resize(PT[i,:,:], (64*13, 256))
        #         mr = cv2.resize(MR[i,:,:], (64*13, 256))
        #     else:
        #         ac = cv2.resize(PT[i,:,:], (1248, 384))
        #         mr = cv2.resize(MR[i,:,:], (1248, 384))
        #     # ac = 255.0 * (1 - ac / np.max(ac))
        #
        #     comb = np.concatenate((np.transpose(ac),
        #                            np.transpose(mr),
        #                            ), 1)
        #     cv2.imwrite(coronal_dir + '/' + prefix_name
        #                 + '_%04d.png'%(i-210), comb.astype(np.uint16))
        #
        # # Sagittal slices
        # sagittal_dir = self.save_dir + 'sagittal'
        # if not os.path.exists(sagittal_dir): os.makedirs(sagittal_dir)
        # for i in range(1, 100):
        #     print('Sagittal', prefix_name, i)
        #     if resize:
        #         ac = cv2.resize(PT[:,i,:], (64*13, 256))
        #         mr = cv2.resize(MR[:,i,:], (64*13, 256))
        #     else:
        #         ac = cv2.resize(PT[:,i,:], (1248, 384))
        #         mr = cv2.resize(MR[:,i,:], (1248, 384))
        #
        #     # ac = 255.0 * (1 - ac / np.max(ac))
        #
        #     comb = np.concatenate((np.transpose(ac),
        #                            np.transpose(mr),
        #                            ), 1)
        #     cv2.imwrite(sagittal_dir + '/' + prefix_name
        #                 + '_%04d.png'%(i-170), comb.astype(np.uint16))
class WB_CT_MR_data():
    """
    Registration between PET and MR data sequnce
    """
    def __init__(self):
        self.prefix_path = 'F:/PET-MR-TRANS/RAW_DATA' #原数据存储地址
        self.save_dir = './data/2021-06-13/' #处理后数据保存地址

    def array_of_case(self, dir, select=None, reverse=False):
        """

        :param dir: directory of source data
        :param select: truncation of source data
        :param reverse:negative film
        :return:data series after proccessing
        """
        files = os.listdir(dir)
        if reverse: files.reverse()
        arrays = []
        for f in files:
            p = os.path.join(dir, f)
            I = read_file(p).pixel_array
            arrays.append(I)
        array = np.array(arrays)
        if select: array = array[select]
        return np.transpose(array, [1, 2, 0]) # transpose input array into regular order

    def save_3d_array(self, folder='001'):
        """
        :param folder: subfolder in the source data directory, it depends on the situation in each cases
        :return:image after registrarion stored as '.npz' temporary
        This function first resize two sequence into same size( both x,y,z )
         then crop them from the central point.
        """
        CT = self.prefix_path + '/' + folder + '/CT'
        PMR = self.prefix_path + '/' + folder + '/MR'
        # PT  = self.array_of_case(PET, reverse=True)
        PT  = self.array_of_case(CT, list(range(1,137)))#concatence
        PT = central_crop(PT, (350,350))

        MR = self.array_of_case(PMR, list(range(1,137)))#range wiped out unneccesarry part e.g. top of skull

        # MR = self.array_of_case(PMR, reverse=True)
        MR = central_crop(MR, (320, 320))
        MR = squarized(MR)
        #range wiped out unneccesarry part e.g. top of skull

        print('Finish loading', PT.shape, MR.shape)
        # PT = zoom(PT, (328/256, 328/256, 164/227)) #first
        PT = zoom(PT, (1, 1, 1))#first
        # three parameter each corresponding to zooming ratio on each direction
        # depends on different case, the ratio can be learnt from radiant automatic zooming operation
        MR = zoom(MR, (1, 1, 1))

        print('Finish resizing', PT.shape, MR.shape)
        # PT = central_crop(PT, (261,261))
        # MR = central_crop(MR, (306,306))

        print('Finish cropping', PT.shape, MR.shape)
        np.savez(self.prefix_path + folder + '_data.npz',
                 PT = PT, MR = MR)

        print('Finish saving', folder + '_data.npz')

    def generate_2d_images(self, file, resize=False):
        # load data
        data = np.load(self.prefix_path + file)
        prefix_name, _ = file.split('_')
        PT, MR = data['PT'], data['MR']
        PT = np.clip(PT, 0, 12000)
        # normalization
        # MR = 255.0 * np.clip((MR / 250.0), 0, 1)

        # Axial slices
        axial_dir = self.save_dir + 'axial'
        print(axial_dir)
        if not os.path.exists(axial_dir):
            os.makedirs(axial_dir)
        for i in range(1, 162):
            print('Axial', prefix_name, i)
            if resize:
                ac = cv2.resize(PT[:,:,i], (256,256))
                mr = cv2.resize(MR[:,:,i], (256,256))
            else:
                ac = PT[:,:,i]
                mr = MR[:,:,i]
            # if i > 85: ac = np.clip(ac, 0, 5000)
            # ac = 255.0 * (1 - ac / np.max(ac))
            comb = np.concatenate((ac, mr), 1)
            # comb = -comb
            cv2.imwrite(axial_dir + '/' + prefix_name + '_%04d.png'
                        %i, comb.astype(np.uint16))
if __name__ == '__main__':
    # generate_PET_MRI_data()

    # c = TB_PET_CT_data()
    # c.save_3d_array('008')
    # for i in range(3,21):
        # c.save_3d_array('%03d'%i)
        # c.generate_2d_images('%03d_data.npz'%i)
    i = 40
    m = WB_CT_MR_data()
    m.save_3d_array('%03d'%i)
    m.generate_2d_images('%03d_data.npz'%i, resize=True)
    # i=35 #case number
    #
    # m.save_3d_array('%03d'%i)
    # m.generate_2d_images('%03d_data.npz'%i, resize=True)
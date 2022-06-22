import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np, cv2, os, random, shutil
epi_img = nib.load('G:/ZJU/LaiSNSE Lab/2020.09-2021/T2data/psy039/BG-T2-severe.nii.gz')
epi_img_data = epi_img.get_data()
epi_img_data.shape
print(epi_img_data.shape)

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
        # slice_0 = epi_img_data[i, :, :]
        # slice_1 = epi_img_data[:, i, :]
        cv2.imwrite('G:/ZJU/LaiSNSE Lab/2020.09-2021/T2data/psy039/test/' + '_%04d.png'
                        % i, slice.T)

slice_2 = epi_img_data
        # show_slices([slice_0, slice_1, slice_2])
show_slices(slice_2)




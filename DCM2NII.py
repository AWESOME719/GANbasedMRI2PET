import SimpleITK as sitk

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames('F:\\ZJU\\LaiSNSE Lab\\2020.09-2021\\T2data\\psy037\\30_Pan')
reader.SetFileNames(dicom_names)
image2 = reader.Execute()
image_array = sitk.GetArrayFromImage(image2) # z, y, x
origin = image2.GetOrigin() # x, y, z
spacing = image2.GetSpacing() # x, y, z
image3=sitk.GetImageFromArray(image2)##其他三维数据修改原本的数据，
sitk.WriteImage(image3,'test.nii') #这里可以直接换成image2 这样就保存了原来的数据成了nii格式了

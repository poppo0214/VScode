import pydicom
import os
import vtk
import itk
from vtkmodules.util import numpy_support
import numpy as np
import glob
from utils_vtk import numpy_to_vtk, save_vtk

if __name__ == '__main__':
  # dcm_path = "D:\\takahashi_k\\dicom\\patient2"
  # filename = "\\IM_0002"
  # out_path = "D:\\takahashi_k\\testDICOM\\patient2"
  # In_Dcm = os.path.join(dcm_path + filename)
  # Out_origin = os.path.join(out_path + filename + "-origin.vti")
  #Out_doppler = os.path.join(out_path + filename + "-doppler.vti")
  make_mhd = True       #mhdを生成するか
  dcm_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\DICOM"
  vti_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\VTI"
  mhd_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\MHD"
  patient_name = r""
  in_path = os.path.join(dcm_path+patient_name)
  rootlist = glob.glob(f"{in_path}/*.dcm")

  for root in rootlist:
    filename = root.replace(in_path, "")
    filename = filename.replace(".dcm", "")
    out_vti = os.path.join(vti_path + "\\"  +  filename + "-Origin.vti")
    print(root)
    print(out_vti)
    data = pydicom.dcmread(root)
    # date = data.InstanceCreationDate
    # equipment = data.ManufacturerModelName
    # patient = data.PatientName
    
    x = int(data.Columns)
    y = int(data.Rows)
    z = int(data.NumberOfFrames)
    spacing = data[0x200d, 0x3303].value
    spacing = str(spacing)
    spacing = spacing.lstrip("['")
    spacing = spacing.rstrip("']")
    spacing = spacing.split("', '")
    for i in range(len(spacing)):
      spacing[i] = float(spacing[i])
    origin = [0, 0, 0]                          #画像原点がなぜかdicomヘッダファイルから取り出せないので適当に設定

    #配列取得
    img_array = data.pixel_array
    #transpose（軸入れ替え）
    img_array = np.transpose(img_array, (2, 1, 0))

    # origin_array = img_array[0:int(z/2), :, :]
    # dopper_array = img_array[int(z/2):z+1, :, :]
    #print(origin_array.shape, dopper_array.shape)

    vti_data = numpy_to_vtk(img_array, spacing, origin)
    #doppler_data = numpy_to_vtk(dopper_array, spacing, origin)

    save_vtk(vti_data, out_vti)
    #save_vtk(doppler_data, Out_doppler)
    if make_mhd:
      Out_mhd = os.path.join(mhd_path + "\\" + filename + "-Origin.mhd")
      #img_array = img_array.reshape()
      mhd_data = itk.GetImageFromArray(img_array)
      mhd_data.SetSpacing(spacing)
      mhd_data.SetOrigin(origin)

      # save output file(.mhd)
      itk.imwrite(mhd_data, Out_mhd)









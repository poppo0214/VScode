"""同じ患者名のdicomデータを同じフォルダに分類する"""
import pydicom
import os
import vtk
from vtkmodules.util import numpy_support
import numpy as np
import glob

if __name__ == '__main__':
  dcm_path = "D:\\takahashi_k\\DICOM"
  rootlist = glob.glob(f"{dcm_path}/*")

  for root in rootlist:
    filename = root.replace(dcm_path, "")

    #患者名取得→フォルダ名としてフォルダ作成
    data = pydicom.dcmread(root) 
    patient = str(data.PatientName)
    print(patient, type(patient))
    folder = os.path.join(dcm_path + "\\" + patient)
    
    # ディレクトリが存在しない場合、ディレクトリを作成する
    if not os.path.exists(folder):
        os.makedirs(folder)
    root_rev = os.path.join(folder + filename + ".dcm")
    os.rename(root, root_rev)


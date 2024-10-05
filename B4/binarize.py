import numpy as np
import itk
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import cv2
import glob

def binarizaiton(array, th):
    print(th)
    compare = array > th
    output = compare.astype(np.int8)
    return(output)
    

if __name__ == '__main__':
    In_path = r'D:\takahashi_k\temporary space'
    bnth_list = [20,30]
    for bnth in bnth_list:
        os.makedirs(In_path, exist_ok=True)

    rootlist = glob.glob(f'{In_path}/*.vti')
    for root in rootlist:
        array, spa, ori = vtk_data_loader(root)
        filename = root.replace(In_path, "")
        filename = filename.replace(".vti", "")

        for bnth in bnth_list:
            output = binarizaiton(array, bnth)
          
            Out_root = os.path.join(In_path + filename + f"_bn({bnth}).vti")
            print(Out_root)
            output = numpy_to_vtk(output, spa, ori)
            save_vtk(output, Out_root)

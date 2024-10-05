import numpy as np
import itk
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import cv2
import glob

def binarizaiton(array):
    th = 10
    compare = array > th
    output = compare.astype(np.int8)
    return(output)


if __name__ == '__main__': 
    """2値化すること!!!!!!!!!"""   
    filtered_path = r'd:\takahashi_k\PaperData\AS(6)_all(1-11)'
    out_path = r"d:\takahashi_k\PaperData\AS(6)_all(1-11)_masked"
    origin_path = r'd:\takahashi_k\database\us\origin_all'

    rootlist = glob.glob(f'{filtered_path}/*.vti')
    for root in rootlist:
        filtered_array, spa, ori = vtk_data_loader(root)
        filename = root.replace(filtered_path, "")
        filename = filename.replace(".vti", "")
        num = filename.find('Origin')
        personname = filename[0:num]

        print(personname, num)

        origin_root = os.path.join(origin_path + personname + "Origin.vti")
        origin_array, spa, ori = vtk_data_loader(origin_root)
        print(origin_root)

        mask_array = binarizaiton(origin_array)
        output = filtered_array * mask_array
        out_root = os.path.join(out_path + filename + "_masked.vti")
        print(out_root)
        output = numpy_to_vtk(output, spa, ori)
        save_vtk(output, out_root)


                



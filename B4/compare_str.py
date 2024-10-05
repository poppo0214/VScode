import numpy as np
import itk
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import cv2
import glob

"""2値化すること!!!!!!!!!"""
if __name__ == '__main__':
    segment_path = r'd:\takahashi_k\database\us\annotation_all'
    filter_path = r'd:\takahashi_k\PaperData\wall(1-11)'

    segment_list = glob.glob(f'{segment_path}/*.vti')
    filter_list = glob.glob(f'{filter_path}/*.vti')
    for segment_root in segment_list:
        match = False
        segment_name = segment_root.replace(segment_path, "")
        segment_name = segment_name.replace("-seg.vti", "")
        
        for filter_root in filter_list:
            if segment_name in filter_root:
                match = True
        if match == False:
            print(f"{segment_name} is not exist!!!")
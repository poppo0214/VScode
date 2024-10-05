import os
import vtk
from vtkmodules.util import numpy_support
import numpy as np
import glob
from utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk

if __name__ == '__main__':
    In_path = r"d:\takahashi_k\database\us\senior\annotation\vti"
    Out_path = r"d:\takahashi_k\database\us\senior\annotation\vti(0-255)"
    
    rootlist = glob.glob(f'{In_path}/*-seg.vti')

    for root in rootlist:
        print(root)
        array, spa, ori = vtk_data_loader(root)
        #[0, 1] -> [0, 255]
        array = array * 255
        #[0, 255] -> [0, 1]
        #array = array / 255

        filename = root.replace(In_path, "")
        Out_root = os.path.join(Out_path + filename)
        output = numpy_to_vtk(array, spa, ori)
        save_vtk(output, Out_root)
        
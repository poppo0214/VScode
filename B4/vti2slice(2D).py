import os
import cv2
import sys
from utils.utils_vtk import vtk_data_loader
import subprocess
inpath = "d:\\takahashi_k\\database\\us\\senior\\origin"
outpath = r"d:\takahashi_k\new_function\us(slice)\both"
infile = "Watanabe_0096_Origin"
inroot = os.path.join(inpath + "\\" + infile + ".vti")
paraview = r"C:\Program Files\ParaView 5.8.1-Windows-Python3.7-msvc2015-64bit\bin\paraview.exe"
openvti = False

if __name__ == '__main__':
    #paraviewを起動し、スライスを選択
    if openvti:
        subprocess.Popen([paraview, inroot])
    print("choose slise")
    axis = input("input axis>>>")
    num = int(input("input slzice number>>>"))

    outslice = os.path.join(outpath + "\\" + infile + f"_slice({axis}{num}).png")

    array, spa, ori = vtk_data_loader(inroot) 
    if axis == "x":
        slice = array[num, :, :]
    elif axis == "y":
        slice = array[:, num, :]
    elif axis == "z":
        slice = array[:, :, num]
    else:
        print("error")
        sys.exit()
    
    cv2.imwrite(outslice, slice)
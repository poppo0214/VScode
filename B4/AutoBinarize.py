import itk
import os
import numpy as np
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import glob
import re
import subprocess as sp

inpath = "D:\\takahashi_k\\frangiedData"
outpath = "D:\\takahashi_k\\binarizedData"
#namelist = ["araki", "ichikawa", "ito", "makino", "okadome", "saito", "shimizu", "watanabe"]
namelist = ['patient3\\part2']
txtroot = os.path.join(outpath + "\\thb(patient).txt")             #txtファイルを作っておくこと
paraviewroot = 'C:\\Program Files\\ParaView 5.8.1-Windows-Python3.7-msvc2015-64bit\\bin\\paraview.exe'
def binarization(root):
    #input vti-data
    root = str(root)
    print("loading vti file...")
    print(root)
    array, spa, ori = vtk_data_loader(root)
    
    #binarization
    print("binarization")
    sp.Popen([paraviewroot, root])
    th = input("th for binarization = ")
    if th == None:
        return
    elif "." in th:
        th = float(th)
    else:
        th = int(th)
    compare = array > th
    output = compare.astype(np.int8)
    
    #save as vti
    print("saving as vti...")
    filename = re.sub(r"D:\\takahashi_k\\frangiedData\\", "", root)
    filename = re.sub('_frg_trim_vp(rev)(OTSU).vti', '', filename)
    f.write(f"{filename}\t{th}\n")

    output = numpy_to_vtk(output, spa, ori)
    outroot = re.sub("frangied", "binarized", root)
    outroot = re.sub(".vti", "", outroot)
    outroot = os.path.join(outroot + "_bn.vti")
    print(outroot)
    save_vtk(output, outroot)
    return
    

if __name__ == '__main__':
    f = open(txtroot, 'a')
    for name in namelist:
        f.write(f"{name}\n")
        path = os.path.join(inpath + "\\" + name)
        print(path)
        rootlist = glob.glob(f'{path}/*_frg_trim_vp(rev)(OTSU).vti')

        for root in rootlist:
            binarization(root)
    # path = os.path.join(inpath + "\\patient3\\part2\\")
    # for i in range(51, 57, 2):
    #     if i < 10:
    #         num = "000" + f"{i}"
    #         root = os.path.join(path + num + "_frg_trim_vp(rev)(OTSU).vti")
    #         binarization(root)

    #     elif i < 100:
    #         num = "00" + f"{i}"
    #         root = os.path.join(path + num + "_frg_trim_vp(rev)(OTSU).vti")
    #         binarization(root)

    #     else:
    #         num = "0" + f"{i}"
    #         root = os.path.join(path + num + "_frg_trim_vp(rev)(OTSU).vti")
    #         binarization(root)

    f.close

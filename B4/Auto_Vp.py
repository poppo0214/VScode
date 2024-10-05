import os
import numpy as np
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import glob
import subprocess as sp
from scipy.ndimage import label
from scipy import ndimage
import re
import itk

inpath = "D:\\takahashi_k\\binarizedData\\byYukino"
outpath = "D:\\takahashi_k\\volumeProcessedData\\byYukino"
namelist = ["araki", "ichikawa", "itou", "makino", "okadome", "saito", "shimizu", "watanabe"]
txtroot = os.path.join(outpath + "\\thForVp.txt")             #txtファイルを作っておくこと


def main(root):
    #input vti-data
    print("loading vti file...")
    root = str(root)
    print(root)
    array, spa, ori = vtk_data_loader(root)

    #make mhd data
    mhd_root = re.sub(".vti", ".mhd", root)
    mhd = itk.GetImageFromArray(array)
    mhd.SetSpacing(spa)
    mhd.SetOrigin(ori)
    itk.imwrite(mhd, mhd_root)

    #volume process
    sp.Popen([r'C:\\Program Files\\ParaView 5.8.1-Windows-Python3.7-msvc2015-64bit\\bin\\paraview.exe', root])
    th = input("th for binarization = ")
    
    #save as vti
    print("saving as vti...")
    f.write(f"{root}\t{th}\n")


if __name__ == '__main__':
    f = open(txtroot, 'a')
    for name in namelist:
        f.write(f"{name}\n")
        path = os.path.join(inpath + "\\" + name)
        print(path)
        rootlist = glob.glob(f'{path}/*_Origin_frg_trim_vp(rev)_bn.vti')

        for root in rootlist:
            main(root)
    f.close


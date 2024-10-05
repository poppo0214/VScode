import os
import numpy as np
import glob
from utils_vtk import vtk_data_loader, vti_to_vtp, save_vtp


if __name__ == '__main__':
    vti_path = r"D:\takahashi_k\registration(model)\forUSE\original\filled"
    vtp_path = r"D:\takahashi_k\registration(model)\forUSE\original\filled"
    if not os.path.exists(vtp_path):
        os.mkdir(vtp_path)
    # filename = "Ogawa_IM_0134-seg_Global_icped_to_Ogawa_IM_0145-seg_Global_1000"
    # inroot = os.path.join(path + "\\" + file_name + ".vtp")
    # out_vti = os.path.join(path + "\\" + file_name + ".vti")

    rootlist = glob.glob(f'{vti_path}/*Y-deform-rev(filled)_padding(20)_closing(10).vti')
    for in_vti in rootlist:
        filename = in_vti.replace(vti_path, "")
        filename = filename.replace(".vti", "")
        out_vtp = os.path.join(vtp_path + filename + ".vtp")
        print(filename)

        #vtiデータ読み込み
        vti_array, spa, ori = vtk_data_loader(in_vti)
        #vti2vtp
        poly_data = vti_to_vtp(vti_array, spa, ori)
        #save polydata
        save_vtp(poly_data, out_vtp)
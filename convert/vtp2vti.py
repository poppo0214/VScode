import os
import numpy as np
from utils_vtk import vtp_data_loader, vtp_to_vti, save_vtk
import glob


if __name__ == '__main__':
    path = r"D:\takahashi_k\registration(expandedBVN)\makino"
    file_name = "makino_mask"
    inroot = os.path.join(path + "\\" + file_name + ".vtp")
    out_vti = os.path.join(path + "\\" + file_name + ".vti")
    reference_root = os.path.join(path + "\\Ogawa_IM_0134-seg.vti")
    print(inroot, out_vti, reference_root)

    #vtpデータ読み込み
    poly_data = vtp_data_loader(inroot)
    #vtp2vti
    image_data = vtp_to_vti(poly_data, spacing_mode="reference", root=reference_root)
    save_vtk(image_data, out_vti)

# if __name__ == '__main__':
#     vtp_path = r"D:\takahashi_k\registration(expandedBVN)\makino"
#     vti_path = r"D:\takahashi_k\registration(expandedBVN)\makino"
#     reference_vtipath = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin"
#     # filename = "Ogawa_IM_0134-seg_Global_icped_to_Ogawa_IM_0145-seg_Global_1000"
#     # inroot = os.path.join(path + "\\" + file_name + ".vtp")
#     # out_vti = os.path.join(path + "\\" + file_name + ".vti")

#     rootlist = glob.glob(f'{vtp_path}/*.vtp')
#     for in_vtp in rootlist:
#         print(in_vtp)
#         poly_data = vtp_data_loader(in_vtp)
#         filename = in_vtp.replace(vtp_path, "")
#         filename = filename.replace(".vtp", "")
#         out_vti = os.path.join(vti_path + filename + ".vti")
#         print(out_vti)
#         filename = filename.replace("-Annotation_Global", "")
#         reference_vtiroot = os.path.join(reference_vtipath + filename + "-Origin.vti")
#         print(reference_vtiroot)

#         #vtpデータ読み込み
#         poly_data = vtp_data_loader(in_vtp)
#         #vti2vtp
#         vti_data = vtp_to_vti(poly_data, "reference", reference_vtiroot)
#         #save polydata
#         save_vtk(vti_data, out_vti)

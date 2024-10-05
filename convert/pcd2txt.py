import numpy as np
import glob
import os
import open3d as o3d

# def list_to_txt(point_list):
#     for i in range(len(point_list)):
#         row = point_list[i]
        
#     return polydata


if __name__ == '__main__':
    # step4. マスク,分割血管網，ソース血管網をそれぞれvti, vtp, pcd形式で出力する
    in_path = r"D:\takahashi_k\registration(differentBVN)\yamaguchi1\RANSAC_ICP"

    rootlist = glob.glob(f'{in_path}/*.pcd')
    for in_root in rootlist:
        in_pcd =  o3d.io.read_point_cloud(in_root)
        in_array = np.asarray(in_pcd.points)
        out_root = in_root.replace(".pcd", "txt")
        np.savetxt(out_root, in_array, delimiter='\t')

    
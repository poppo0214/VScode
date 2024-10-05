import os
import numpy as np
import glob
from utils_vtk import vtk_data_loader
import open3d as o3d

def vti_to_pcd(vti_array, spacing, origin):
    points_list = []

    # 3次元バイナリデータをスキャンしてポイントクラウドを生成
    dimensions = vti_array.shape
    print(dimensions)
    for z in range(dimensions[2]):  # Z方向
        for y in range(dimensions[1]):  # Y方向
            for x in range(dimensions[0]):  # X方向
                # 各座標のバイナリ値を取得
                value = vti_array[x, y, z]
                # バイナリ値が1ならポイントを追加
                if value >= 1:
                    #print(value)
                    # 座標とスペーシングに基づいて点の位置を計算
                    point_x = origin[0] + x * spacing[0]
                    point_y = origin[1] + y * spacing[1]
                    point_z = origin[2] + z * spacing[2]

                    points_list.append([point_x, point_y, point_z])
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_list)

    return(pcd)


if __name__ == '__main__':
    vti_path = r"D:\takahashi_k\registration(model)\forUSE\original\filled"
    pcd_path = r"D:\takahashi_k\registration(model)\forUSE\original\filled"
    if not os.path.exists(pcd_path):
        os.mkdir(pcd_path)
    # filename = "Ogawa_IM_0134-seg_Global_icped_to_Ogawa_IM_0145-seg_Global_1000"
    # inroot = os.path.join(path + "\\" + file_name + ".vtp")
    # out_vti = os.path.join(path + "\\" + file_name + ".vti")

    rootlist = glob.glob(f'{vti_path}/*Y-deform-rev(filled)_padding(20)_closing(10).vti')
    for in_vti in rootlist:
        filename = in_vti.replace(vti_path, "")
        filename = filename.replace(".vti", "")
        out_vtp = os.path.join(pcd_path + filename + ".pcd")
        print(filename)

        #vtiデータ読み込み
        vti_array, spa, ori = vtk_data_loader(in_vti)
        #vti2vtp
        pcd_data = vti_to_pcd(vti_array, spa, ori)
        #save polydata
        print(f"saving...\n{out_vtp}")
        o3d.io.write_point_cloud(out_vtp, pcd_data)        
import numpy as np
import vtk
from utils.utils_vtk import vtk_data_loader, vtp_to_vti, save_vtk, save_vtp
import os
import open3d as o3d
import glob
"""複数のvtiを結合し，vti, vtp, pcdファイルで出力する"""

def aiton(array, th):
    print(th)
    compare = array > th
    output = compare.astype(np.int8)
    return(output)

def vti_to_pointlist(vti_array, spacing, origin):
    """二値化されたvtiから1の座標を取り出しリストに収納する"""
    # ポイントクラウドのための空のリストを作成
    point_list = []
    # points = vtk.vtkPoints()                   
    # vertices = vtk.vtkCellArray()

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
                    point_list.append([point_x, point_y, point_z])

                    #point = points.InsertNextPoint(point_x, point_y, point_z)
                    # vertices.InsertNextCell(1)
                    # vertices.InsertCellPoint(point)
                    
                    # polydata = vtk.vtkPolyData()
                    # polydata.SetPoints(points)
                    # polydata.SetVerts(vertices)


    return(point_list)

def list_to_vtp(point_list):
    points = vtk.vtkPoints()                   
    vertices = vtk.vtkCellArray()

    for i in range(len(point_list)):
        point = points.InsertNextPoint(point_list[i])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
    return polydata



if __name__ == '__main__':
    root = r"D:\takahashi_k\database\us\expanded\shimizu\notuse"
    #namelist = ["expansionVolume-2_0", "expansionVolume-2_1", "expansionVolume-2_2"]
    namelist = ["expansionVolume-0_0", "expansionVolume-0_1", "expansionVolume-0_2", "expansionVolume-0_3"]
    #namelist = ["expansionVolume-0_0", "expansionVolume-0_1"]
    outname = "shimizu0"
    outvtp_root = os.path.join(root + "\\" + outname + "(filled).vtp")
    outpcd_root = os.path.join(root + "\\" + outname + "(filled).pcd")
    outvti_root = os.path.join(root + "\\" + outname + "(filled).vti")

    allpoint_list = []
    spa_sum = [0, 0, 0]

    for name in namelist:    
        inroot = os.path.join(root + "\\" + name + ".vti")
        print(inroot)
        img_array, spa, ori = vtk_data_loader(inroot)
    
        point_list = vti_to_pointlist(img_array, spa, ori)
        spa_sum = [spa_sum[0]+spa[0], spa_sum[1]+spa[1], spa_sum[2]+spa[2]]
        allpoint_list += point_list
        print(type(allpoint_list))
        print(sum(len(v) for v in allpoint_list))
           

    outpcd = o3d.geometry.PointCloud()
    outpcd.points = o3d.utility.Vector3dVector(allpoint_list)    
    o3d.io.write_point_cloud(outpcd_root, outpcd)

    outvtp = list_to_vtp(allpoint_list)
    save_vtp(outvtp, outvtp_root)  
    
    spa_ave = [spa_sum[0]/len(namelist), spa_sum[1]/len(namelist), spa_sum[2]/len(namelist)]
    print(spa_ave)
    outvti = vtp_to_vti(outvtp, "manual", outvti_root, spa_ave)
    save_vtk(outvti, outvti_root)


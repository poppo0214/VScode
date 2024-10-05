import numpy as np
import vtk
from utils.utils_vtk import vtk_data_loader, vtp_to_vti, save_vtk, save_vtp
import os
import open3d as o3d
import glob
"""複数のpcdを結合し，vti, vtp, pcdファイルで出力する"""

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
    regied_path = r"D:\takahashi_k\registration(differentBVN)\yamaguchi1\RANSAC_ICP"
    target_path = r"D:\takahashi_k\registration(differentBVN)\yamaguchi1"
    target_name = "IM_0328_target"
    out_path = os.path.join(target_path + "\\combied")
    spa = [0.5, 0.5, 0.5]
    mode = "glob"
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    target_root = os.path.join(target_path + "\\" + target_name + ".pcd")
    target_pcd = o3d.io.read_point_cloud(target_root)
    target_array = np.asarray(target_pcd.points)

    regiedlist = glob.glob(f'{regied_path}/*.pcd')
    for regied_root in regiedlist:
        regied_pcd =  o3d.io.read_point_cloud(regied_root)
        regied_array = np.asarray(regied_pcd.points)

        out_name = regied_root.replace(regied_path, "")
        out_name = out_name.replace("_target.pcd", "")
        out_name = out_name.replace("to", "and")

        outvtp_root = os.path.join(out_path + "\\" + out_name + ".vtp")
        outpcd_root = os.path.join(out_path + "\\" + out_name + ".pcd")
        outvti_root = os.path.join(out_path + "\\" + out_name + ".vti")

        combinedArray =  np.concatenate([regied_array, target_array])
        
        outpcd = o3d.geometry.PointCloud()
        outpcd.points = o3d.utility.Vector3dVector(combinedArray)
        outpcd = outpcd.voxel_down_sample(voxel_size=0.0001)
        o3d.io.write_point_cloud(outpcd_root, outpcd)

        combinedPoint_list = np.asarray(outpcd.points)
        outvtp = list_to_vtp(combinedPoint_list)
        save_vtp(outvtp, outvtp_root)  
        
        outvti = vtp_to_vti(outvtp, "manual", outvti_root, spa)
        save_vtk(outvti, outvti_root)



# if __name__ == '__main__':
#     path = r"D:\takahashi_k\registration(expandedBVN)\makino\RANSAC_ICP"    
#     spa = [0.5, 0.5, 0.5]
#     mode = "glob"
#     #mode = "specify"
#     namelist = ["0043-0035-0031(surface)", "0041(surface)_10-0_Regied_to_0043-0035-0031(surface)"]
#     #namelist = ["expansionVolume-0_0", "expansionVolume-0_1", "expansionVolume-0_2", "expansionVolume-0_3"]
#     #namelist = ["expansionVolume-0_0", "expansionVolume-0_1"]
#     outname = "combine\\0043-0035-0031-0041(surface)"

#     outvtp_root = os.path.join(path + "\\" + outname + ".vtp")
#     outpcd_root = os.path.join(path + "\\" + outname + ".pcd")
#     outvti_root = os.path.join(path + "\\" + outname + ".vti")

#     if mode == "glob":
#         rootlist = glob.glob(f'{path}/*.pcd')
    
#     elif mode == "specify":
#         rootlist = []
#         for name in namelist:
#             root = os.path.join(path + "\\" + name + ".pcd")
#             rootlist.append(root)

#     combinedArray = np.empty((1,3))
#     for root in rootlist:
#         print(root)
#         #print(combinedArray.shape)
#         pcd = o3d.io.read_point_cloud(root)
#         tempArray = np.asarray(pcd.points)
#         #print(tempArray.shape)
#         combinedArray = np.concatenate([combinedArray, tempArray])
        
#     outpcd = o3d.geometry.PointCloud()
#     outpcd.points = o3d.utility.Vector3dVector(combinedArray)
#     outpcd = outpcd.voxel_down_sample(voxel_size=0.0001)
#     o3d.io.write_point_cloud(outpcd_root, outpcd)

#     combinedPoint_list = np.asarray(outpcd.points)
#     outvtp = list_to_vtp(combinedPoint_list)
#     save_vtp(outvtp, outvtp_root)  
    
#     outvti = vtp_to_vti(outvtp, "manual", outvti_root, spa)
#     save_vtk(outvti, outvti_root)


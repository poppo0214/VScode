import os
import numpy as np
from utils.utils_vtk import vtp_data_loader, save_vtp
import glob
import vtk
import random
import math
import open3d as o3d
# """複数のデータに対して1つの値で並進移動指せる場合"""
# if __name__ == '__main__':
#     Inpath = r"D:\takahashi_k\database\us\divide\divide(^1_4)"
#     Outpath = r"D:\takahashi_k\database\us\divide\devide(^1_4)_move(10)"
#     """ターゲット側は動かさなくてよい！！！"""
#     Error = 10               #どれだけずらすか[mm]

#     rootlist = glob.glob(f'{Inpath}/*2-*.vtp')
#     for Inroot in rootlist:
#         #vtpデータ読み込み
#         InPoly = vtp_data_loader(Inroot)
#         filename = Inroot.replace(Inpath, "")
#         filename = filename.replace(".vtp", "")
#         print(filename)

#         #各軸の並進移動量をランダムに決定する
#         x_ran = random.random()
#         y_ran = random.random()
#         z_ran = random.random()
#         D = math.sqrt(x_ran*x_ran + y_ran*y_ran + z_ran*z_ran)
#         x_error = x_ran*Error/D
#         y_error = y_ran*Error/D
#         z_error = z_ran*Error/D
#         D = math.sqrt(x_error*x_error + y_error*y_error + z_error*z_error)
#         print(D)

#         num = InPoly.GetNumberOfPoints()            #vtpのポイント数を取得
#         points = vtk.vtkPoints()                 
#         vertices = vtk.vtkCellArray()

#         for i in range(num):
#             point = np.array(InPoly.GetPoint(i))
#             #print(point)
#             #new_point = [point[0]-x_error, point[1]-y_error, point[2]-z_error]
#             #point_list.append(new_point)
#             new_point = points.InsertNextPoint(point[0]-x_error, point[1]-y_error, point[2]-z_error)
#             #print(new_point)
#             vertices.InsertNextCell(1)
#             vertices.InsertCellPoint(new_point)
            
#             polydata = vtk.vtkPolyData()
#             polydata.SetPoints(points)
#             polydata.SetVerts(vertices)

#         out_vtp = os.path.join(Outpath + "\\" + filename + f"_{Error}[{x_error}, {y_error}, {z_error}].vtp")
#         save_vtp(polydata, out_vtp)

#"""1つのデータに対して複数の値で並進移動させる場合"""
# if __name__ == '__main__':
#     Inpath = r"D:\takahashi_k\database\us\devide\YAMAGUCHI_0328\divide"
#     Outpath = r"D:\takahashi_k\database\us\devide\YAMAGUCHI_0328\divied(1_8)-move(1-10)"
#     file_name = "IM_0328-Annotation_2-5"
#     Inroot = os.path.join(Inpath + "\\" + file_name + ".vtp")
#     #vtpデータ読み込み
#     InPoly = vtp_data_loader(Inroot)
#     filename = Inroot.replace(Inpath, "")
#     filename = filename.replace(".vtp", "")
#     print(filename)

#     max_Error = 30               #どれだけずらすか[mm]
#     for Error in range(11, max_Error+1, 1):        
#         #各軸の並進移動量をランダムに決定する
#         x_ran = random.random()
#         y_ran = random.random()
#         z_ran = random.random()
#         D = math.sqrt(x_ran*x_ran + y_ran*y_ran + z_ran*z_ran)
#         x_error = x_ran*Error/D
#         y_error = y_ran*Error/D
#         z_error = z_ran*Error/D
#         D = math.sqrt(x_error*x_error + y_error*y_error + z_error*z_error)
#         print(D)

#         num = InPoly.GetNumberOfPoints()            #vtpのポイント数を取得
#         points = vtk.vtkPoints()                 
#         vertices = vtk.vtkCellArray()

#         for i in range(num):
#             point = np.array(InPoly.GetPoint(i))
#             #print(point)
#             #new_point = [point[0]-x_error, point[1]-y_error, point[2]-z_error]
#             #point_list.append(new_point)
#             new_point = points.InsertNextPoint(point[0]-x_error, point[1]-y_error, point[2]-z_error)
#             #print(new_point)
#             vertices.InsertNextCell(1)
#             vertices.InsertCellPoint(new_point)
            
#             polydata = vtk.vtkPolyData()
#             polydata.SetPoints(points)
#             polydata.SetVerts(vertices)

#         out_vtp = os.path.join(Outpath + "\\" + filename + f"_{Error}[{x_error}, {y_error}, {z_error}].vtp")
#         save_vtp(polydata, out_vtp)


# """複数のデータを複数の値で移動させる場合(vtp)"""
# if __name__ == '__main__':
#     Inpath = r"D:\takahashi_k\database\us\divide\divide(1_8)"
#     Outpath = r"D:\takahashi_k\database\us\divide\divide(1_8)_move(1-30)"
#     """ターゲット側は動かさなくてよい！！！"""
#     max_Error = 30               #どれだけずらすか[mm]

#     rootlist = glob.glob(f'{Inpath}/*2-*.vtp')
#     for Inroot in rootlist:
#         #vtpデータ読み込み
#         InPoly = vtp_data_loader(Inroot)
#         filename = Inroot.replace(Inpath, "")
#         filename = filename.replace(".vtp", "")
#         print(filename)

#         for Error in range(1, max_Error+1, 1):        
#             #各軸の並進移動量をランダムに決定する
#             x_ran = random.random()
#             y_ran = random.random()
#             z_ran = random.random()
#             D = math.sqrt(x_ran*x_ran + y_ran*y_ran + z_ran*z_ran)
#             x_error = x_ran*Error/D
#             y_error = y_ran*Error/D
#             z_error = z_ran*Error/D
#             D = math.sqrt(x_error*x_error + y_error*y_error + z_error*z_error)
#             print(D)

#             num = InPoly.GetNumberOfPoints()            #vtpのポイント数を取得
#             points = vtk.vtkPoints()                 
#             vertices = vtk.vtkCellArray()

#             for i in range(num):
#                 point = np.array(InPoly.GetPoint(i))
#                 #print(point)
#                 #new_point = [point[0]-x_error, point[1]-y_error, point[2]-z_error]
#                 #point_list.append(new_point)
#                 new_point = points.InsertNextPoint(point[0]-x_error, point[1]-y_error, point[2]-z_error)
#                 #print(new_point)
#                 vertices.InsertNextCell(1)
#                 vertices.InsertCellPoint(new_point)
                
#                 polydata = vtk.vtkPolyData()
#                 polydata.SetPoints(points)
#                 polydata.SetVerts(vertices)

#             out_vtp = os.path.join(Outpath + "\\" + filename + f"_{Error}[{x_error}, {y_error}, {z_error}].vtp")
#             save_vtp(polydata, out_vtp)

"""複数のデータを複数の値で移動させる場合(pcd)"""
if __name__ == '__main__':
    Inpath = r"D:\takahashi_k\registration\original\pcd"
    Outpath = r"D:\takahashi_k\registration\translation11"
    if not os.path.exists(Outpath):
        os.mkdir(Outpath)
    """ターゲット側は動かさなくてよい！！！"""
    Error_list = range(0, 31, 5)              #どれだけずらすか[mm]

    rootlist = glob.glob(f'{Inpath}/*.pcd')
    for Inroot in rootlist:
        #vtpデータ読み込み,numpy配列に変換
        pcd_data = o3d.io.read_point_cloud(Inroot)
        #o3d.visualization.draw_geometries([pcd_data])
        xyz_array = np.asarray(pcd_data.points)
        filename = Inroot.replace(Inpath, "")
        filename = filename.replace(".pcd", "")
        print(filename)

        for Error in Error_list:        
            #各軸の並進移動量をランダムに決定する
            x_ran = random.random()
            y_ran = random.random()
            z_ran = random.random()
            D = math.sqrt(x_ran*x_ran + y_ran*y_ran + z_ran*z_ran)
            x_error = x_ran*Error/D
            y_error = y_ran*Error/D
            z_error = z_ran*Error/D
            D = math.sqrt(x_error*x_error + y_error*y_error + z_error*z_error)
            print(D)

            error_array = np.array([x_error, y_error, z_error])
            xyz_array = xyz_array - error_array
            
            #numpy配列を# numpy をOpen3d に変換
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz_array)
            out_pcd = os.path.join(Outpath + "\\" + filename + f"_{str(Error).zfill(2)}[{x_error}, {y_error}, {z_error}].pcd")
            o3d.io.write_point_cloud(out_pcd, pcd)        
            print(f"saving...\n{out_pcd}")
           
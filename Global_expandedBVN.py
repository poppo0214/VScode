"""拡張済み血管網をtargetのポラリスからの同次変換行列をもちいてグローバル座標にするコード
入力：拡張済み血管網（vti）, ターゲットの同次変換行列（mtx）
出力：pcd, vtp, vti"""

import numpy as np
import vtk
from utils.utils_vtk import save_vtp, vtk_data_loader, vtp_to_vti, save_vtk
import glob
import open3d as o3d
import csv
import os

def read_mtx(mtxroot):
    """スペース区切りとかいうキショイデータ形式から同次変換行列だけ取り出してnumpy配列にする"""
    mtx_list = []
    with open(mtxroot, encoding='utf-8', newline='') as f:
        csv_reader = csv.reader(f, delimiter="\t")
        for row in csv_reader:
            #print(row)
            if len(row) == 1 and row[0][0] != "%":
                row = row[0]
                row = row.replace(",", "")
                row = row.split()
                mtx_list.append(row)
    mtx = np.asarray(mtx_list)
    mtx = mtx.astype(np.float64)

    return mtx

def PointCloud_mtx(point_array, mtx):
    paded_array = np.pad(point_array,((0,0),(0,1)),  mode="constant", constant_values=1)
    moved_list = []
    for i in range(paded_array.shape[0]):
        origin_point = paded_array[i, :].T
        moved_point = np.dot(mtx, origin_point)
        moved_list.append(moved_point)
        #print(origin_point, moved_point)
    moved_array = np.asarray(moved_list)[:, :3]
    return moved_array

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

def main(expanded_root, filename, target_root, moved_path):
    mtx = read_mtx(target_root)
    expanded_vtiArray, spa, ori = vtk_data_loader(expanded_root)

    expanded_PointList = vti_to_pointlist(expanded_vtiArray, spa, ori)
    
    moved_PointList = PointCloud_mtx(expanded_PointList, mtx)
    moved_vtp = list_to_vtp(moved_PointList)
    vtp_moved_root = os.path.join(moved_path + "\\" + filename + "-moved.vtp")
    save_vtp(moved_vtp, vtp_moved_root)

    pcd_moved_root = os.path.join(moved_path + "\\" + filename + "-moved.pcd")
    moved_pcd = o3d.geometry.PointCloud()
    moved_pcd.points = o3d.utility.Vector3dVector(moved_PointList)    
    o3d.io.write_point_cloud(pcd_moved_root, moved_pcd)

    vti_moved_root = os.path.join(moved_path + "\\" + filename + "-moved.vti")
    moved_vti =  vtp_to_vti(moved_vtp, "manual", vtp_moved_root, spa)
    save_vtk(moved_vti, vti_moved_root)




if __name__ == '__main__':   
    expanded_path = r"D:\takahashi_k\database\us\expanded\shimizu"
    expanded_name = "shimizu"
    target_path = r"D:\takahashi_k\database\us\expanded\shimizu\matrix"
    target_name = "IM_0025_Phantom2Volume"
    moved_path = expanded_path

    expanded_root = os.path.join(expanded_path + "\\" + expanded_name + ".vti")
    target_root = os.path.join(target_path + "\\" + target_name + ".mtx")
    main(expanded_root, expanded_name, target_root, moved_path)

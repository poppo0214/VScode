import numpy as np
import vtk
from utils.utils_vtk import vtp_to_vti, vtk_data_loader, numpy_to_vtk, save_vtk, save_vtp
import os
import open3d as o3d
import glob
import csv

def make_mask(origin_array):
    mask_array = origin_array > 0
    mask_array = mask_array.astype(np.int8)
    return mask_array

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

def PointCloud_mtx(point_array, mtx):
    print(point_array.shape)
    print(mtx)
    print(mtx.shape)
    paded_array = np.pad(point_array,((0,0),(0,1)),  mode="constant", constant_values=1)
    moved_list = []
    for i in range(paded_array.shape[0]):
        origin_point = paded_array[i, :].T
        moved_point = np.dot(mtx, origin_point)
        moved_list.append(moved_point)
        #print(origin_point, moved_point)
    moved_array = np.asarray(moved_list)[:, :3]
    return moved_array

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


def main(origin_path, mask_path, mtx_path):
    
    origin_list = glob.glob(f'{origin_path}/*.vti')
    for origin_root in origin_list:
        # step1
        # Originを二値化，マスクボリュームとする
        origin_array, spa, ori = vtk_data_loader(origin_root)
        mask_array = make_mask(origin_array)
        
        filename = origin_root.replace(origin_path, "")
        filename = filename.replace("-Origin.vti", "")
        print(filename)
        vtimask_root = os.path.join(mask_path + "\\" + filename + "-mask.vti")
        mask_vti = numpy_to_vtk(mask_array, spa, ori)
        save_vtk(mask_vti, vtimask_root)

        # step2
        # マスクボリュームを点群化
        mask_PointList = vti_to_pointlist(mask_array, spa, ori)
        mask_vtp = list_to_vtp(mask_PointList)
        vtp_mask_root = os.path.join(mask_path + "\\" + filename + "-mask.vtp")
        save_vtp(mask_vtp, vtp_mask_root)

        pcd_mask_root = os.path.join(mask_path + "\\" + filename + "-mask.pcd")
        mask_pcd = o3d.geometry.PointCloud()
        mask_pcd.points = o3d.utility.Vector3dVector(mask_PointList)    
        o3d.io.write_point_cloud(pcd_mask_root, mask_pcd)

        # step3
        # 同次変換行列でグローバル座標系に置き直す
        # mtxファイル読み込み
        mtxroot = os.path.join(mtx_path + "\\" + filename + "_Phantom2Volume.mtx")
        mtx = read_mtx(mtxroot)        

        #　グローバル座標に直して出力
        mask_PointArray = np.asarray(mask_PointList)
        movedMask_Pointlist = PointCloud_mtx(mask_PointArray, mtx)
        movedMask_vtp = list_to_vtp(movedMask_Pointlist)
        vtp_movedMask_root = os.path.join(mask_path + "\\" + filename + "-movedMask.vtp")
        save_vtp(movedMask_vtp, vtp_movedMask_root)

        vti_movedMask_root = os.path.join(mask_path + "\\" + filename + "-movedMask.vti")
        movedMask_vti = vtp_to_vti(movedMask_vtp, "manual", vti_movedMask_root, spa)
        save_vtk(movedMask_vti, vti_movedMask_root)

        pcd_movedMask_root = os.path.join(mask_path + "\\" + filename + "-movedMask.pcd")
        movedMask_pcd = o3d.geometry.PointCloud()
        movedMask_pcd.points = o3d.utility.Vector3dVector(movedMask_Pointlist)    
        o3d.io.write_point_cloud(pcd_movedMask_root, movedMask_pcd)
        



if __name__ == '__main__':    
    origin_path = r"D:\takahashi_k\database\us\expanded\filled\makino\origin"
    mask_path = r"D:\takahashi_k\database\us\expanded\filled\makino\mask"
    mtx_path = r"D:\takahashi_k\database\us\expanded\filled\makino\matrix"

    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    
    main(origin_path, mask_path, mtx_path)
    
    
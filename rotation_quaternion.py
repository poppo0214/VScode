import os
import numpy as np
from utils.utils_vtk import vtp_data_loader, save_vtp
import glob
import vtk
import random
import math
import open3d as o3d
import quaternion
import copy

def draw_registration_result(source, target):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries([source_temp, target_temp])

def pcd_mtx(pcd_array, mtx):
    paded_array = np.pad(pcd_array,((0,0),(0,1)),  mode="constant", constant_values=1)
    moved_list = []
    for i in range(paded_array.shape[0]):
        origin_point = paded_array[i, :].T
        moved_point = np.dot(mtx, origin_point)
        moved_list.append(moved_point)
        #print(origin_point, moved_point)
    moved_array = np.asarray(moved_list)[:, :3]
    return moved_array

def Rotation_quat(pcd_array, theta_deg):
    theta_rad = math.radians(theta_deg )
    vx = random.random()
    vy = random.random()
    vz = random.random()
    D = math.sqrt(vx*vx + vy*vy + vz*vz)
    vx = vx*1/D
    vy = vy*1/D
    vz = vz*1/D
    D = math.sqrt(vx*vx + vy*vy + vz*vz)
    print(D)   
    quat = np.quaternion(math.cos(theta_rad/2), vx*math.sin(theta_rad/2), vy*math.sin(theta_rad/2), vz*math.sin(theta_rad/2))
    rot_matrix = np.asarray(quaternion.as_rotation_matrix(quat))
    print(rot_matrix)
    #roted_pcd = pcd_data.rotate(rot_matrix)
    trs_matrix = np.pad(rot_matrix,((0,1),(0,1)),  mode="constant", constant_values=0)
    trs_matrix[3,3] = 1
    roted_array = pcd_mtx(pcd_array, trs_matrix)
    return roted_array, trs_matrix


"""複数のデータを複数の値で移動させる場合(pcd)"""
if __name__ == '__main__':
    Inpath = r"D:\takahashi_k\registration(model)\forUSE\original"
    Outpath = Inpath
    """回転軸はランダムで、回転角のみ指定できる"""
    theta_list = range(10, 31, 10)
    N = 10                                   #何データ作成するか

    rootlist = glob.glob(f'{Inpath}/*-0.pcd')
    for Inroot in rootlist:
        #pcdデータ読み込み,numpy配列に変換
        pcd_data = o3d.io.read_point_cloud(Inroot)
        #o3d.visualization.draw_geometries([pcd_data])
        pcd_array = np.asarray(pcd_data.points)
        filename = Inroot.replace(Inpath, "")
        filename = filename.replace(".pcd", "")
        print(filename)

        for theta in theta_list:
            Outpcd_path = os.path.join(Outpath + f"\\rotation_{theta}deg")
            if not os.path.exists(Outpcd_path):
                os.mkdir(Outpcd_path)
            Outmtx_path = os.path.join(Outpcd_path + f"\\TrsMtx_GT")
            if not os.path.exists(Outmtx_path):
                os.mkdir(Outmtx_path)
        
            for num in range(0, N):        
                TrsMtx_root = os.path.join(Outmtx_path + f"\\{filename}_{theta}-{num}.tsv")           
                Outpcd_root = os.path.join(Outpcd_path + f"\\{filename}_{theta}-{num}.pcd")

 
                moved_array, trs_mtx = Rotation_quat(pcd_array, theta)
                #draw_registration_result(pcd_data, moved_pcd)

                print(trs_mtx)
                # #numpy配列を# numpy をOpen3d に変換
                moved_pcd = o3d.geometry.PointCloud()
                moved_pcd.points = o3d.utility.Vector3dVector(moved_array)
                np.savetxt(TrsMtx_root, trs_mtx, delimiter='\t')
                o3d.io.write_point_cloud(Outpcd_root, moved_pcd) 
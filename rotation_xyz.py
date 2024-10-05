import os
import numpy as np
from utils.utils_vtk import vtp_data_loader, save_vtp
import glob
import vtk
import random
import math
import open3d as o3d

def Rotation_xyz(pointcloud, theta_x, theta_y, theta_z):
    theta_x = math.radians(theta_x)
    theta_y = math.radians(theta_y)
    theta_z = math.radians(theta_z)
    rot_x = np.array([[ 1,                 0,                  0],
                      [ 0, math.cos(theta_x), -math.sin(theta_x)],
                      [ 0, math.sin(theta_x),  math.cos(theta_x)]])

    rot_y = np.array([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                      [                 0, 1,                  0],
                      [-math.sin(theta_y), 0, math.cos(theta_y)]])

    rot_z = np.array([[ math.cos(theta_z), -math.sin(theta_z), 0],
                      [ math.sin(theta_z),  math.cos(theta_z), 0],
                      [                 0,                  0, 1]])

    rot_matrix = rot_z.dot(rot_y.dot(rot_x))
    rot_pointcloud = rot_matrix.dot(pointcloud.T).T
    return rot_pointcloud, rot_matrix

"""複数のデータを複数の値で移動させる場合(pcd)"""
if __name__ == '__main__':
    Inpath = r"D:\takahashi_k\registration\translation"
    Outpath = r"D:\takahashi_k\registration\rotation"
    """ターゲット側は動かさなくてよい！！！"""
    x_angle = range(0, 181, 30)               #どれだ回転させるか[degree]
    y_angle = range(0, 181, 30)               #どれだ回転させるか[degree]
    z_angle = range(0, 181, 30)               #どれだ回転させるか[degree]

    rootlist = glob.glob(f'{Inpath}/*.pcd')
    for Inroot in rootlist:
        #vtpデータ読み込み,numpy配列に変換
        pcd_data = o3d.io.read_point_cloud(Inroot)
        #o3d.visualization.draw_geometries([pcd_data])
        xyz_array = np.asarray(pcd_data.points)
        filename = Inroot.replace(Inpath, "")
        filename = filename.replace(".pcd", "")
        print(filename)

        for x_degree in x_angle:
            for y_degree in y_angle:
                for z_degree in z_angle:        
                    #回転させる
                    after_array = Rotation_xyz(xyz_array, x_degree, y_degree, z_degree)
                    
                    #numpy配列を# numpy をOpen3d に変換
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(xyz_array)
                    out_pcd = os.path.join(Outpath + "\\" + filename + f"_[{x_degree}, {y_degree}, {z_degree}].pcd")
                    o3d.io.write_point_cloud(out_pcd, pcd)        
                    print(f"saving...\n{out_pcd}")
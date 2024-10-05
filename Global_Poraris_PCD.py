import open3d as o3d
import numpy as np
import copy
import os
import glob
import time

"""https://whitewell.sakura.ne.jp/Open3D/ICPRegistration.html"""
"""https://www.open3d.org/docs/latest/tutorial/t_pipelines/t_icp_registration.html"""
"""https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html"""


### Helper visualization function
def draw_registration_result(probe, target, transformation):
    probe_temp = copy.deepcopy(probe)
    target_temp = copy.deepcopy(target)
    probe_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    probe_temp.transform(transformation)
    o3d.visualization.draw_geometries([probe_temp, target_temp])

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



if __name__ == "__main__":
    """pathの設定  mtxファイルの拡張子を.txtに変更すること"""
    path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI"
    Probe_pcdpath = os.path.join(path + "\\Annotation\\PCD(surface)")              #入力のプローブ座標系のpcdデータ
    Global_pcdpath = os.path.join(path + "\\Global\\PCD(surface)")      #出力のグローバル座標系のpcdデータ
    mtxpath = os.path.join(path + "\\Matrix")

    if not os.path.exists(Global_pcdpath):
        os.mkdir(Global_pcdpath)

    rootlist = glob.glob(f'{Probe_pcdpath}/*.pcd')
    num = 1
    for pcd_root in rootlist:
     
        probe_pcd = o3d.io.read_point_cloud(pcd_root)
        probe_array = np.asarray(probe_pcd.points)
        filename = pcd_root.replace(Probe_pcdpath, "")
        filename = filename.replace(".pcd", "")
        mtxroot = os.path.join(mtxpath + f"\\Volume{num}.csv")
        print(mtxroot)
        Global_pcdroot = os.path.join(Global_pcdpath + filename + "_Global.pcd")

        #同時変換行列を読み込み
        # mtx = np.loadtxt(mtxroot)
        # print(mtx)
        f = open(mtxroot, 'r')
        mtx = f.read()
        mtx = mtx.replace(",", "")
        mtx = mtx.split()
        mtx = np.array(mtx, dtype=float)
        mtx = mtx.reshape(4,4)
        print(mtx)
        print(type(mtx))
        f.close()

        Global_array = pcd_mtx(probe_array, mtx)
        Global_pcd = o3d.geometry.PointCloud()
        Global_pcd.points = o3d.utility.Vector3dVector(Global_array)
        o3d.io.write_point_cloud(Global_pcdroot, Global_pcd)
        num += 1
                

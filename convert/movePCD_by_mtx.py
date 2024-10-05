import os
from pycpd import DeformableRegistration
import numpy as np
import open3d as o3d

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
    # pcd_path = r"D:\takahashi_k\registration(model)\forUSE\RANSAC+ICP\Y-target(surfacev2)"
    # pcd_name = r"Y-source(surfacev2)"
    # #mtx_path = pcd_path
    # mtx_path = r"D:\takahashi_k\registration(model)\forUSE\RANSAC+ICP\Y-target(surfacev2)\TrsMtx_EM"
    # mtx_name = r"Y-source(filled)_Regied_to_Y-target(filled)"
    # out_path = pcd_path
    # out_name = pcd_name + "_moved"

    pcd_path = r"D:\takahashi_k\registration(model)\forUSE\source"
    pcd_name = r"Y-source-rev(surface)"
    #mtx_path = pcd_path
    mtx_path = r"D:\takahashi_k\registration(model)\forUSE\RANSAC+ICP\Y-target(filled)\TrsMtx_EM"
    mtx_name = r"Y-source-rev(filled)_Regied_to_Y-target(filled)"
    out_path = r"D:\takahashi_k\registration(model)\forUSE\RANSAC+ICP\Y-target(surface)"
    out_name = pcd_name + "_moved"

    pcd_root = os.path.join(pcd_path + "\\" + pcd_name + ".pcd")
    print(pcd_root)
    mtx_root = os.path.join(mtx_path + "\\" + mtx_name + ".tsv")
    print(mtx_root)
    out_root = os.path.join(out_path + "\\" + out_name + ".pcd")
    print(out_root)

    pcd_data = o3d.io.read_point_cloud(pcd_root)
    pcd_array = np.asarray(pcd_data.points)

    f = open(mtx_root, 'r')
    mtx = f.read()
    mtx = mtx.replace(",", "")
    mtx = mtx.split()
    mtx = np.array(mtx, dtype=float)
    mtx = mtx.reshape(4,4)
    print(mtx)
    print(type(mtx))
    f.close()

    moved_array = pcd_mtx(pcd_array, mtx)
    moved_pcd = o3d.geometry.PointCloud()
    moved_pcd.points = o3d.utility.Vector3dVector(moved_array)
    o3d.io.write_point_cloud(out_root, moved_pcd)


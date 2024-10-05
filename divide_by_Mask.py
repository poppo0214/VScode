import numpy as np
import vtk
from utils.utils_vtk import vtk_data_loader, vtp_to_vti, save_vtk, save_vtp
import os
import open3d as o3d
import glob
import csv

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

def main(mask_path, expanded_path, expanded_name, divided_path):
    expanded_pcdroot = os.path.join(expanded_path + "\\" + expanded_name + "-moved.pcd")
    expanded_pcd = o3d.io.read_point_cloud(expanded_pcdroot)
    expanded_pcdTree = o3d.geometry.KDTreeFlann(expanded_pcd)
    expanded_vtiroot = os.path.join(expanded_path + "\\" + expanded_name + "-moved.vti")
    expanded_array, spa, ori = vtk_data_loader(expanded_vtiroot)
    d = (spa[0]*spa[0] + spa[1]*spa[1] + spa[2]*spa[2]) ** (0.5)
    print(d)

    mask_rootlist = glob.glob(f'{mask_path}/*movedMask.pcd')
    for mask_pcdroot in mask_rootlist:
        mask_pcd = o3d.io.read_point_cloud(mask_pcdroot)

        num = mask_pcdroot.replace(mask_path, "")
        num = num.replace("IM_", "")
        num = num.replace("-movedMask.pcd", "")
        num = num.replace("\\", "")
        print(num)
        filename = expanded_name + "-divided" + num  
        mask_pointArray = np.asarray(mask_pcd.points)
        

        masked_pointList =[]
        for num in range(mask_pointArray.shape[0]):
            point = mask_pointArray[num, :]
            [k, idx, _] = expanded_pcdTree.search_radius_vector_3d(point, d)
            if k > 0:
                masked_pointList.append(point)
        print(len(masked_pointList))

        divided_vtp = list_to_vtp(masked_pointList)
        vtp_divided_root = os.path.join(divided_path + "\\" + filename + ".vtp")
        save_vtp(divided_vtp, vtp_divided_root)

        pcd_divided_root = os.path.join(divided_path + "\\" + filename + ".pcd")
        divided_pcd = o3d.geometry.PointCloud()
        divided_pcd.points = o3d.utility.Vector3dVector(masked_pointList)    
        o3d.io.write_point_cloud(pcd_divided_root, divided_pcd)

        vti_divided_root = os.path.join(divided_path + "\\" + filename + ".vti")
        divided_vti =  vtp_to_vti(divided_vtp, "manual", vtp_divided_root, spa)
        save_vtk(divided_vti, vti_divided_root)


if __name__ == '__main__':    
    origin_path = r"D:\takahashi_k\database\us\expanded\filled\makino\origin"
    mask_path = r"D:\takahashi_k\database\us\expanded\filled\makino\mask"
    mtx_path = r"D:\takahashi_k\database\us\expanded\filled\makino\matrix"
    expanded_path = r"D:\takahashi_k\database\us\expanded\filled\makino"
    expanded_name = r"makino"                                         #拡張子なし
    divided_path = r"D:\takahashi_k\database\us\expanded\filled\makino\divided"   

    if not os.path.exists(divided_path):
        os.mkdir(divided_path)
    
    main(mask_path, expanded_path, expanded_name, divided_path)
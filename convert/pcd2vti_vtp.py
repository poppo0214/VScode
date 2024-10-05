import numpy as np
import glob
import vtk
from utils_vtk import vtk_data_loader, vtp_to_vti, save_vtk, save_vtp
import os
import open3d as o3d

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

def output_vtp_vti(array, vtproot, vtiroot, spa):
    print(vtproot)
    print(vtiroot)
    vtp = list_to_vtp(array)
    vti = vtp_to_vti(vtp, "manual", vtiroot, spa)
    save_vtp(vtp, vtproot)
    save_vtk(vti, vtiroot)

if __name__ == '__main__':
    # step4. マスク,分割血管網，ソース血管網をそれぞれvti, vtp, pcd形式で出力する
    in_path = r"D:\takahashi_k\registration(model)\forUSE\RANSAC+ICP\Y-target(surface)"
    #spa = [0.36813678694904567, 0.3157286665464158, 0.5453666041673727]
    spa = [0.5,0.5,0.5]

    rootlist = glob.glob(f'{in_path}/*.pcd')
    for in_root in rootlist:
        in_pcd =  o3d.io.read_point_cloud(in_root)
        in_array = np.asarray(in_pcd.points)

        out_vtiroot = in_root.replace(".pcd", ".vti")
        out_vtproot = in_root.replace(".pcd", ".vtp")

        output_vtp_vti(in_array, out_vtproot, out_vtiroot, spa)

    
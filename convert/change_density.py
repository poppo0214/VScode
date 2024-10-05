import open3d as o3d
import numpy as np
import vtk
import os
from utils_vtk import save_vtp

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

def output_vtp(array, vtproot):
    vtp = list_to_vtp(array)
    save_vtp(vtp, vtproot)


if __name__ == "__main__":
    inpath = r"D:\takahashi_k\registration(model)\forUSE\original\surface"
    outpath = r"D:\takahashi_k\registration(model)\forUSE\original\surface"
    filename = "Y_padding(20)_closing(10)(surface)"
    N=10
    output_as_vtp = True

    out_pcd = os.path.join(outpath + "\\" + filename + f"(1_{N}).pcd")
    
    inroot = os.path.join(inpath + "\\" + filename + ".pcd")
    inpcd= o3d.io.read_point_cloud(inroot)
    inarray = np.asarray(inpcd.points)

    origin_num = inarray.shape[0]
    out_array = inarray[::N]
    down_num = out_array.shape[0]
    print(origin_num, down_num, down_num/origin_num)

    outroot = os.path.join(outpath + "\\" + filename + f"(1_{N}).pcd")
    outpcd = o3d.geometry.PointCloud()
    outpcd.points = o3d.utility.Vector3dVector(out_array)
    o3d.io.write_point_cloud(outroot, outpcd)

    if output_as_vtp:
        out_vtp = out_pcd.replace(".pcd", ".vtp")
        output_vtp(out_array, out_vtp)


    



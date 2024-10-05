import numpy as np
import glob
import vtk
from utils_vtk import vtk_data_loader, vtp_to_vti, save_vtk, save_vtp, vtk_to_numpy
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

def GetData_fromVti(vti_root):    
    print(vti_root)
    #get data from mhdfile
    vtk_reader = vtk.vtkXMLImageDataReader()
    vtk_reader.SetFileName(vti_root)
    vtk_reader.Update()
    vtk_data = vtk_reader.GetOutput()

    spacing = list(vtk_data.GetSpacing())
    origin = list(vtk_data.GetOrigin())
    extent = list(vtk_data.GetExtent())
    bounds = list(vtk_data.GetBounds())
    array = vtk_to_numpy(vtk_data).astype(np.float32)
    dim = list(array.shape)
    print("spacing: ", spacing, "\norigin", origin, "\nextent", extent, "\ndim", dim, "\nbounds", bounds)
    return (spacing, origin, dim, extent, bounds)


# if __name__ == '__main__':
#     dir_path = r"D:\takahashi_k\registration(differentBVN)\YAMAGUCHI\RANSAC-ICP2"
#     Origin_path = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\VTI_rev'
#     #mode = "same_dir"                   #dirpath と同じ階層にvti, vtpの階層を作る
#     mode = "down_dir"                   #dirpath の1つ下の階層にpcdを移動させ、それと同じ階層にvti, vtpの階層を作る
    
#     print(f"mode : {mode}")
#     for num in range(328, 341, 1):
#         num = str(num).zfill(4)
#         in_path = os.path.join(dir_path + "\\IM_" + num + "(filled)_Global")
#         print(in_path)

#         rootlist = glob.glob(f'{in_path}/*.pcd')
#         for in_root in rootlist:
#             in_pcd =  o3d.io.read_point_cloud(in_root)
#             in_array = np.asarray(in_pcd.points)

#             if mode == "same_dir":
#                 filename = in_root.replace(in_path, "")
#                 filename = filename.replace(".pcd", "")
                
#                 out_vtipath = in_path.replace("PCD", "VTI")
#                 out_vtppath = in_path.replace("PCD", "VTP")

            
#             elif mode == "down_dir":
#                 filename = in_root.replace(in_path, "")
#                 in_newpath = os.path.join(in_path + "\\PCD")
#                 print(in_newpath)
#                 if not os.path.exists(in_newpath):
#                     os.mkdir(in_newpath)
#                 os.rename(in_root, os.path.join(in_newpath + filename))    

#                 filename = filename.replace(".pcd", "")
#                 out_vtipath = in_newpath.replace("PCD", "VTI")
#                 out_vtppath = in_newpath.replace("PCD", "VTP")
            
#             print(out_vtipath)
#             print(out_vtppath)

#             if not os.path.exists(out_vtipath):
#                 os.mkdir(out_vtipath)
#             if not os.path.exists(out_vtppath):
#                 os.mkdir(out_vtppath)
#             out_vtiroot = os.path.join(out_vtipath + "\\" + filename + ".vti")
#             out_vtproot = os.path.join(out_vtppath + "\\" + filename + ".vtp")            
            
            
#             """RANSAC+ICP後のPCDの場合"""
#             filename = "IM_" + filename.split("_")[1]
#             print(filename)
#             """"""            

            
#             # print(out_vtiroot)
#             # print(out_vtproot)

#             filename = (filename.split("(")[0]).replace("(", "")
#             #print(filename)
#             origin_root = os.path.join(Origin_path + "\\" + filename + "-Origin.vti")
#             print(origin_root)
#             spacing, origin, dim, extent, bounds = GetData_fromVti(origin_root)
#             output_vtp_vti(in_array, out_vtproot, out_vtiroot, spacing)

if __name__ == '__main__':
    in_path = r"D:\takahashi_k\registration(differentBVN)\YAMAGUCHI\CPD\target"
    Origin_path = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\VTI_rev'

    out_vtipath = in_path.replace("PCD", "VTI")
    out_vtppath = in_path.replace("PCD", "VTP")
    if not os.path.exists(out_vtipath):
        os.mkdir(out_vtipath)
    if not os.path.exists(out_vtppath):
        os.mkdir(out_vtppath)

    rootlist = glob.glob(f'{in_path}/*.pcd')
    for in_root in rootlist:
        in_pcd =  o3d.io.read_point_cloud(in_root)
        in_array = np.asarray(in_pcd.points)
        
        filename = in_root.replace(in_path, "")
        filename = filename.replace(".pcd", "")

        out_vtiroot = os.path.join(out_vtipath + "\\" + filename + ".vti")
        out_vtproot = os.path.join(out_vtppath + "\\" + filename + ".vtp")
        # print(out_vtiroot)
        # print(out_vtproot)

        filename = (filename.split("(")[0]).replace("(", "")
        #print(filename)
        origin_root = os.path.join(Origin_path + "\\" + filename + "-Origin.vti")
        #print(origin_root)
        spacing, origin, dim, extent, bounds = GetData_fromVti(origin_root)
        output_vtp_vti(in_array, out_vtproot, out_vtiroot, spacing)

    
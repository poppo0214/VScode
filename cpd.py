import os
from pycpd import DeformableRegistration
import numpy as np
import open3d as o3d
import numpy as np
import glob
import vtk
from utils.utils_vtk import vtp_to_vti, save_vtk, save_vtp, vtk_to_numpy

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:, 0],  X[:, 1], X[:, 2], color='red', label='Target')
    ax.scatter(Y[:, 0],  Y[:, 1], Y[:, 2], color='blue', label='Source')
    ax.text2D(0.87, 0.92, 'Iteration: {:d}'.format(
        iteration), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)



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

if __name__ == "__main__":
    source_path = r"D:\takahashi_k\registration(model)\forUSE\RANSAC+ICP\Y-target(surface)"
    source_name = r"Y-source-rev(surface)_moved"
    #target_path = source_path
    target_path = r"D:\takahashi_k\registration(model)\forUSE\target"
    target_name = r"Y-target(surfacev2)"
    out_path = r"D:\takahashi_k\registration(model)\forUSE\CPD\surface"
    spa = [0.5, 0.5, 0.5]


    source_root = os.path.join(source_path + "\\" + source_name + ".pcd")
    print(source_root)
    target_root = os.path.join(target_path + "\\" + target_name + ".pcd")
    print(target_root)

    source = o3d.io.read_point_cloud(source_root)
    target = o3d.io.read_point_cloud(target_root)
    source_array = np.asarray(source.points)
    target_array = np.asarray(target.points)
    # source_array = np.asarray(source.points)[::10]
    # target_array = np.asarray(target.points)[::10]

    # print(f"source:({source_array.shape})", source_array[:, 0:5])
    # print(f"target:({target_array.shape})", target_array[:, 0:5])

    # X=target_array
    # Y=source_array    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # callback = partial(visualize, ax=ax)
    # reg = DeformableRegistration(**{'X': X, 'Y': Y})
    #reg.register(callback)
    #plt.show()

    reg = DeformableRegistration(X=target_array, Y=source_array)
    result = reg.register()
    #print(f"result\t len:{len(result)}\n", result)
    result1 = result[0]
    # result2 = result[1][0]
    # result3 = result[1][1]
    print(f"result1({result1.shape}):", result1[:, 0:10])
    # print(f"result2({result2.shape}):", result2[:, 0:10])
    # print(f"result3({result3.shape}):", result3[:, 0:10])
   
    out_pcdroot = os.path.join(out_path + f"\\{source_name}_cpded_to_{target_name}.pcd" )
    out_vtiroot = os.path.join(out_path + f"\\{source_name}_cpded_to_{target_name}.vti" )
    out_vtproot = os.path.join(out_path + f"\\{source_name}_cpded_to_{target_name}.vtp" )

    outpcd = o3d.geometry.PointCloud()
    outpcd.points = o3d.utility.Vector3dVector(result1)
    o3d.io.write_point_cloud(out_pcdroot, outpcd)

    output_vtp_vti(result1, out_vtproot, out_vtiroot, spa)


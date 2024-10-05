import glob
import vtk
import itk
import os
import sys
import math
from vtkmodules.util import numpy_support
import numpy as np
import open3d as o3d
import pydicom
#import vtkmodules.vtkInteractionStyle
#import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOImage import vtkNIFTIImageWriter
from vtkmodules.vtkImagingStencil import (
    vtkImageStencil, vtkPolyDataToImageStencil)

def stl_to_surface_pcd(stl_root, spacing):
    print(f"loading...\n{stl_root}")
    #stl_data = mesh.Mesh.from_file(filled_root)
    mesh_data = o3d.io.read_triangle_mesh(stl_root)
    vertices = np.asarray(mesh_data.vertices)
    triangles = np.asarray(mesh_data.triangles)
    filename = stl_root.replace(stl_path, "")
    filename = filename.replace(".stl", "")
    print(filename)

    #元の点群のboundsを計算
    x_max = np.max(vertices[:, 0:1])
    x_min = np.min(vertices[:, 0:1])
    y_max = np.max(vertices[:, 1:2])
    y_min = np.min(vertices[:, 1:2])
    z_max = np.max(vertices[:, 2:])        
    z_min = np.min(vertices[:, 2:])
    
    #boundsとspacingからextentsを計算
    x_extent = round((x_max-x_min)/spacing[0])
    y_extent = round((y_max-y_min)/spacing[1])
    z_extent = round((z_max-z_min)/spacing[2])
    print(f"x\tfrom {x_min} to {x_max}: {x_extent}")
    print(f"y\tfrom {y_min} to {y_max}: {y_extent}")
    print(f"z\tfrom {z_min} to {z_max}: {z_extent}")

    #全体の体積とmeshの数から、mesh上に配置する点の数を決める
    volume = (x_max-x_min)*(y_max-y_min)*(z_max-z_min)
    point_num = round(volume/triangles.shape[0])
    # if point_num > 20:
    #     point_num = 10
    print(f"number of points which generated on a mesh:{point_num}")

    #mesh上に点を追加、各meshについてやる
    print("complimenting surface...")
    new_points = []
    for i in range(triangles.shape[0]):
        triangle = triangles[i]           #triangle＝vertices（頂点）のインデックスのようなもの
        vertice1 = vertices[triangle[0]]
        vertice2 = vertices[triangle[1]]
        vertice3 = vertices[triangle[2]]
        # OP = sOA + tOB + uOC; s + t + u = 1; s>=0; t>=0; u>=0
        for a in range(0,point_num):
            flag = True                             #abcが全部0なら点を追加しない
            if a == 0:
                a = sys.float_info.min
                #print(a)
            else:
                flag = False
            for b in range(0,point_num):
                if b == 0:
                    b = sys.float_info.min
                    #print(b)
                else:
                    flag = False
                
                for c in range(0,point_num):
                    if c == 0:
                        c = sys.float_info.min    
                    s = a/(a+b+c)
                    t = b/(a+b+c)
                    u = c/(a+b+c)
                    new_point = s*vertice1 + t*vertice2 + u*vertice3
                    new_points.append(new_point)

                    
    #meshの頂点のポイントと、計算したmesh上のポイントを結合し、pcdデータに変換
    # surface_points = np.concatenate([vertices, np.asarray(new_points)])
    surface_points = np.asarray(new_points)
    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(surface_points)

    return(surface_pcd)

def vtk_to_numpy(data):
    """
    This function is to transform vtk to numpy
    Args
        data: vtk data
    Return: numpy data
    """
    temp = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
    dims = data.GetDimensions()
    global spa
    spa = data.GetSpacing()
    component = data.GetNumberOfScalarComponents()
    if component == 1:
        numpy_data = temp.reshape(dims[2], dims[1], dims[0])
        numpy_data = numpy_data.transpose(2,1,0)
    elif component == 3 or component == 4:
        if dims[2] == 1: # a 2D RGB image
            numpy_data = temp.reshape(dims[1], dims[0], component)
            numpy_data = numpy_data.transpose(0, 1, 2)
            numpy_data = np.flipud(numpy_data)
        else:
            raise RuntimeError('unknow type')
    return numpy_data

def stl_to_vti(stl_root, spacing, origin):
    reader = vtkSTLReader()
    reader.SetFileName(stl_root)
    reader.Update()
    model = reader.GetOutput()

    # generate image
    whiteImage = vtkImageData()
    bounds = model.GetBounds()
    #spacing = [0.5, 0.5, 0.5]
    whiteImage.SetSpacing(spacing)

    dim = [0, 0, 0]
    for i in range(0, 3):
        dim[i] = math.ceil((bounds[i*2+1]-bounds[i*2])/spacing[i])
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)

    origin = [0, 0, 0]
    origin[0] = bounds[0] + spacing[0] / 2
    origin[1] = bounds[2] + spacing[1] / 2
    origin[2] = bounds[4] + spacing[2] / 2

    whiteImage.SetOrigin(origin)
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    count = whiteImage.GetNumberOfPoints()
    for i in range(0, count):
        whiteImage.GetPointData().GetScalars().SetTuple1(i, 1)

    # polygonal data --> image stencil:
    pol2stnc = vtkPolyDataToImageStencil()
    pol2stnc.SetInputData(model)
    pol2stnc.SetOutputOrigin(origin)
    pol2stnc.SetOutputSpacing(spacing)
    pol2stnc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stnc.Update()

    # cut the corresponding white image and set the background:
    imgstenc = vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilConnection(pol2stnc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()
    vti_data = imgstenc.GetOutput()

    return(vti_data)
    

def stl_reader(stl_root):
    reader = vtkSTLReader()
    reader.SetFileName(stl_root)
    reader.Update()
    stl_data = reader.GetOutput()
    return stl_data

def compute_dim(stl_root,spacing):
    #元の点群のboundsを計算
    mesh_data = o3d.io.read_triangle_mesh(stl_root)
    vertices = np.asarray(mesh_data.vertices)
    x_max = np.max(vertices[:, 0:1])
    x_min = np.min(vertices[:, 0:1])
    y_max = np.max(vertices[:, 1:2])
    y_min = np.min(vertices[:, 1:2])
    z_max = np.max(vertices[:, 2:])        
    z_min = np.min(vertices[:, 2:])
    
    #boundsとspacingからextentsを計算
    x_dim = round((x_max-x_min)/spacing[0])
    y_dim = round((y_max-y_min)/spacing[1])
    z_dim = round((z_max-z_min)/spacing[2])

    return(x_dim, y_dim, z_dim)


def vti_to_pcd(vti_array, spacing, origin):
     # ポイントクラウドのための空のリストを作成
    points = []

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

                    points.append([point_x, point_y, point_z])
                    #print([point_x, point_y, point_z])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return(pcd)


def save_pcd(pcd, out_put_path):
    o3d.io.write_point_cloud(out_put_path, pcd)        
    print(f"saving...\n{out_put_path}")



def main(origin_path, stl_path, output_path, spacing_type, spacing=[0.5, 0.5, 0.5]):
    filled_pcd_path = os.path.join(output_path + "\\PCD(filled)")
    surface_pcd_path = os.path.join(output_path + "\\PCD(surface)")
    if not os.path.exists(filled_pcd_path):
        os.mkdir(filled_pcd_path)
    if not os.path.exists(surface_pcd_path):
        os.mkdir(surface_pcd_path)

    rootlist = glob.glob(f'{stl_path}/*.stl')
    for stl_root in rootlist:
        stl_data = stl_reader(stl_root)
        "ファイル名取得"
        filename = stl_root.replace(stl_path, "")
        filename = filename.replace(".stl", "")        
        filename = filename.replace("-Annotation", "")
        filename = filename.replace("\\", "")
        filled_pcd_root = os.path.join(filled_pcd_path + "\\" + filename + "(filled).pcd")
        surface_pcd_root = os.path.join(surface_pcd_path + "\\" + filename + "(surface).pcd")
        print(filename)

        "spacing取得"
        if spacing_type == "VTI":
            in_vtk_root = os.path.join(origin_path + "\\" + filename + "-Origin.vti")
            vtk_reader = vtk.vtkXMLImageDataReader()
            vtk_reader.SetFileName(in_vtk_root)
            vtk_reader.Update()
            data = vtk_reader.GetOutput()
            #dims = data.GetDimensions()
            spacing = data.GetSpacing()
            #ori = data.GetOrigin()

        if spacing_type == "MHD":
            in_vtk_root = os.path.join(origin_path + "\\" + filename + "-Origin.mhd")
            mhd_data = itk.imread(in_vtk_root)
            spacing = mhd_data.GetSpacing()
            #ori = input.GetOrigin()

        print(spacing)
        origin = [0, 0, 0]                          #画像原点がなぜかdicomヘッダファイルから取り出せないので適当に設定
        # x_dim, y_dim, z_dim = compute_dim(stl_root, spacing)
        # dim = [x_dim, y_dim, z_dim]
        # extent = [0, x_dim-1, 0, y_dim-1, 0, z_dim]

        "stl -> surface pcd"
        surface_pcd_data = stl_to_surface_pcd(stl_root, spacing)
        save_pcd(surface_pcd_data, surface_pcd_root)

        "stl->vti（中空が埋まっている）"
        vti_data = stl_to_vti(stl_root, spacing, origin)
        "vti->numpy"
        vti_array = vtk_to_numpy(vti_data)
        "numpy->filled pcd"
        filled_pcd_data = vti_to_pcd(vti_array, spacing, origin)
        save_pcd(filled_pcd_data, filled_pcd_root)

if __name__ == '__main__':
    """spacing
    Dicom:Dicomから取得、diocmは一つ下の階層に入れること
    manual:自分で設定、spacingに渡すこと,もし渡さない場合は[0.5, 0.5, 0.5]になります"""
    origin_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\VTI"
    stl_path = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI\Annotation\STL'             
    output_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Annotation"
    #dicom_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\DICOM"
    spacing_type = "VTI"
    
    main(origin_path, stl_path, output_path, spacing_type)

 





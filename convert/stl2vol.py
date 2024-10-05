#onogi氏が作成、stlからvolumeデータへ、空洞も無し

### Copyright (c) Shinya Onogi
###
import glob
import math
import vtk
import itk
import os
from vtkmodules.util import numpy_support
import numpy as np
#import vtkmodules.vtkInteractionStyle
#import vtkmodules.vtkRenderingOpenGL2
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOImage import vtkNIFTIImageWriter
# from vtkmodules.vtkRenderingCore import (
#     vtkActor,
#     vtkPolyDataMapper,
#     vtkRenderWindow,
#     vtkRenderWindowInteractor,
#     vtkRenderer
# )
from vtkmodules.vtkImagingStencil import (
    vtkImageStencil, vtkPolyDataToImageStencil)

def vtk_to_numpy(data):
  """
  This function is to transform vtk to numpy
  Args
      data: vtk data
  Return: numpy data
  """
  temp = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
  dims = data.GetDimensions()
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


def main_mhd(mhd, filename):
    reader = vtkSTLReader()
    reader.SetFileName(filename + '.stl')
    reader.Update()
    model = reader.GetOutput()

    # generate image
    whiteImage = vtkImageData()

    #get data from mhdfile
    mhdpath = os.path.join(mhd + '.mhd')
    mhddata = itk.imread(mhdpath)
    spacing = mhddata.GetSpacing()
    origin = mhddata.GetOrigin()
    array = itk.GetArrayFromImage(mhddata)
    dim = array.shape
    Ex = dim[2]-1
    Ey = dim[1]-1
    Ez = dim[0]-1

    [Sx, Sy, Sz] = spacing
    [Ox, Oy, Oz] = origin
    print("spacing: ", spacing, "\norigin", origin)
    bounds = [Ox, Ox+(Ex+1)*Sx, Oy, Oy+(Ey+1)*Sy, Oz, Oz+(Ez+1)*Sz]
    
    whiteImage.SetSpacing(spacing)

    dim = [0, 0, 0]
    for i in range(0, 3):
        dim[i] = math.ceil((bounds[i*2+1]-bounds[i*2])/spacing[i])
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)

    whiteImage.SetOrigin(origin)
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    count = whiteImage.GetNumberOfPoints()
    #print(count)
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

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename + '.vti')
    writer.SetInputConnection(imgstenc.GetOutputPort())
    writer.Write()

def main_vti(vti, filename):
    reader = vtkSTLReader()
    reader.SetFileName(filename + '.stl')
    reader.Update()
    model = reader.GetOutput()

    # generate image
    whiteImage = vtkImageData()

    #get data from mhdfile
    vtipath = os.path.join(vti + '.vti')
    vtk_reader = vtk.vtkXMLImageDataReader()
    vtk_reader.SetFileName(vtipath)
    vtk_reader.Update()
    vtk_data = vtk_reader.GetOutput()

    spacing = list(vtk_data.GetSpacing())
    origin = list(vtk_data.GetOrigin())
    print("spacing: ", spacing, "\norigin", origin)

    array = vtk_to_numpy(vtk_data).astype(np.float32)
    dim = array.shape
    Ex = dim[2]-1
    Ey = dim[1]-1
    Ez = dim[0]-1

    [Sx, Sy, Sz] = spacing
    [Ox, Oy, Oz] = origin
    bounds = [Ox, Ox+(Ex+1)*Sx, Oy, Oy+(Ey+1)*Sy, Oz, Oz+(Ez+1)*Sz]
    
    whiteImage.SetSpacing(spacing)

    dim = [0, 0, 0]
    for i in range(0, 3):
        dim[i] = math.ceil((bounds[i*2+1]-bounds[i*2])/spacing[i])
    whiteImage.SetDimensions(dim)
    whiteImage.SetExtent(0, dim[0]-1, 0, dim[1]-1, 0, dim[2]-1)

    whiteImage.SetOrigin(origin)
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    count = whiteImage.GetNumberOfPoints()
    #print(count)
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

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename + '.vti')
    writer.SetInputConnection(imgstenc.GetOutputPort())
    writer.Write()


def main(filename):
    reader = vtkSTLReader()
    reader.SetFileName(filename + '.stl')
    reader.Update()
    model = reader.GetOutput()

    # generate image
    whiteImage = vtkImageData()
    bounds = model.GetBounds()
    spacing = [0.5, 0.5, 0.5]
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

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename + '.vti')
    writer.SetInputConnection(imgstenc.GetOutputPort())
    writer.Write()

def main_rev(stl_root, out_root, spacing, origin):
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

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(out_root)
    writer.SetInputConnection(imgstenc.GetOutputPort())
    writer.Write()


# if __name__ == '__main__':
#     SetOriginSpacing = False
#     mhd = False
#     data_root = r'd:\takahashi_k\database\CT\kobayashi12_annotation'             #stlファイルのルート（拡張子なし）
#     origin_root =  r'd:\takahashi_k\database\us\kobayashi\62\0062-Origin'   #出力画像と座標系をそろえたいファイルのルート（拡張子なし）

#     if SetOriginSpacing:
#         if mhd:
#             main_mhd(origin_root, data_root)
#         else:
#            main_vti(origin_root, data_root)
    
#     else:
#         main(data_root)



# if __name__ == '__main__':
#     stl_path = r'D:\takahashi_k\registration\original'             
#     origin_path = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\VTI_rev'       #出力画像とspacingをそろえたいファイルのルート（拡張子なし）
#     Out_path = r'D:\takahashi_k\registration\original'  

#     rootlist = glob.glob(f'{stl_path}/*.stl')
#     for stl_root in rootlist:
#         filename = stl_root.replace(stl_path, "")
#         #filename = filename.replace("\\Kurihara_IM", "")
#         filename = filename.replace("-Annotation.stl", "")
#         print(filename)        

#         origin_root = os.path.join(origin_path + filename + f"-Origin.vti")
#         spacing
#         Out_root = os.path.join(Out_path + filename + "-Annotation.vti")
#         main_rev(stl_root, origin_root, spacing, origin)


if __name__ == '__main__':
    stl_path = r'D:\takahashi_k\registration(model)\forUSE\original\STL'             
    Out_path = r'D:\takahashi_k\registration(model)\forUSE\original\filled'
    spacing = [0.5, 0.5, 0.5]  
    origin = [0, 0, 0]

    rootlist = glob.glob(f'{stl_path}/*.stl')
    for stl_root in rootlist:
        filename = stl_root.replace(stl_path, "")
        #filename = filename.replace("\\Kurihara_IM", "")
        filename = filename.replace(".stl", "")
        print(filename)
        
        Out_root = os.path.join(Out_path + filename + ".vti")
        main_rev(stl_root, Out_root, spacing, origin)




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
from vtkmodules.vtkImagingStencil import (
    vtkImageStencil, vtkPolyDataToImageStencil, vtkImageStencilToImage)
#from vtk import vvtkImageStencilToImage
from utils_vtk import save_vtk

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

def stl_reader(stl_root):
    reader = vtkSTLReader()
    reader.SetFileName(stl_root)
    reader.Update()
    model = reader.GetOutput()
    return model

def save_vtp(poly, output_path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(poly)
    writer.Write()

def save_vti(img, output_path):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(img)
    writer.Write()

def GetData_fromVti(vti_root):    
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
    print("spacing: ", spacing, "origin", origin, "extent", extent, "dim", dim, "bounds", bounds)
    return (spacing, origin, dim, extent, bounds)

def generate_vti(model, spacing, bounds):
    whiteImage = vtkImageData()
    #bounds = model.GetBounds()
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
    img = imgstenc.GetOutput()

    return img


if __name__ == '__main__':
    """originのvtiとspacingとoriginを合わせる"""
    stl_path = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI\Annotation(0)\STL'             
    Origin_path = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\VTI'
    #Out_vtp = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI^_KAKUTYOU^^^\Annotation\VTP(filled)'        
    Out_vti = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI\Annotation(0)\VTI(filled)'  
    #Out_pcd = r'D:\takahashi_k\database\us\20240502_YAMAGUCHI^_KAKUTYOU^^^\Annotation\PCD(filled)' 
    if not os.path.exists(Out_vti):
        os.mkdir(Out_vti)
    

    rootlist = glob.glob(f'{stl_path}/*.stl')
    for stl_root in rootlist:
        model = stl_reader(stl_root)
        filename = stl_root.replace(stl_path, "")
        filename = filename.replace(".stl", "")
        #vtp_root = os.path.join(Out_vtp + filename + ".vtp")
        vti_root = os.path.join(Out_vti + filename + ".vti")        
        #pcd_root = os.path.join(Out_vti + filename + ".pcd")
        filename = filename.replace("-Annotation", "")
        print(filename)

        origin_root = os.path.join(Origin_path + filename + "-Origin.vti")
        print(stl_root)
        print(origin_root)
        print(vti_root)

        spacing, origin, dim, extent, bounds = GetData_fromVti(origin_root)
        origin = [0,0,0]
        # generate image
        vtidata = generate_vti(model, spacing, bounds)
        vtidata.SetOrigin(origin)
        save_vtk(vtidata, vti_root)







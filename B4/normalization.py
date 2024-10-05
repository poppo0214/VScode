import numpy as np
import itk
import vtk
from vtkmodules.util import numpy_support
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk


inpath = "D:\\takahashi_k\\frangi\\sigma_th"
outpath = "D:\\takahashi_k\\frangi\\sigma_th"
filename = "\\IM_0022-Origin_frg"
mhd = True                 #mhd->True  vti->False


def norm_0_255(array):
    output = array*(255/array.max()) 
    return (output)

if mhd:
    print("loading mhd...")
    In_Img = os.path.join(inpath + filename + ".mhd")
    #image to array
    input = itk.imread(In_Img)
    spa = input.GetSpacing()
    ori = input.GetOrigin()
    array = itk.GetArrayFromImage(input)

else:
    print("loading vti...")
    In_Img = os.path.join(inpath + filename + ".vti")
    #array = vtk_data_loader(In_Img)
    
    vtk_reader = vtk.vtkXMLImageDataReader()

    vtk_reader.SetFileName(In_Img)
    vtk_reader.Update()
    data = vtk_reader.GetOutput()
    dims = data.GetDimensions()
    array = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
    array= array.reshape(dims[2], dims[1], dims[0])

#normalization
output = norm_0_255(array)

if mhd:
    print("saving as mhd...")
    # array to image
    output = itk.GetImageFromArray(output)
    output.SetSpacing(spa)
    output.SetOrigin(ori)

    # save output file(.mhd)
    Out_Img = os.path.join(outpath + filename + "_norm(0-255).mhd")
    itk.imwrite(output, Out_Img)

else:
    print("saving as vti...")
    # array to image
    output = numpy_to_vtk(output)

    # save output file(.vti)
    Out_Img = os.path.join(outpath + filename + "_norm(0-255).vti")
    save_vtk(output, Out_Img)

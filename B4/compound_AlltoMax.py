import numpy as np
import itk
import vtk
from utils.labeling import labeling
from vtkmodules.util import numpy_support
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os

def compound(allarray, maxarray):
    #labeling
    labeled_allarray, Nall = labeling(allarray)
    labeled_maxarray, Nall = labeling(maxarray)

    compoundarray = maxarray

    for N in range(1, Nall+1):
        extracted_image = (labeled_allarray==N)
        if np.sum(extracted_image*labeled_maxarray) != 0:
            compoundarray += extracted_image*allarray
    
    return(compoundarray)


inpath = "D:\\takahashi_k\\frangi\\for_analyse"
outpath = "D:\\takahashi_k\\frangi\\for_analyse"
allfilename = "\\IM_0022-Origin_frg(1-9)_norm(0-255)_trm15vp"
maxfilename = "\\IM_0022-Origin_frg(9)_norm(0-255)_trm15vp"
mhd = True                 #mhd->True  vti->False


if mhd:
    print("loading mhd...")
    all_path = os.path.join(inpath + allfilename + ".mhd")
    max_path = os.path.join(inpath + maxfilename + ".mhd")

    #image to array
    all = itk.imread(all_path)
    max = itk.imread(max_path)
    spa = all.GetSpacing()
    ori = all.GetOrigin()
    allarray = itk.GetArrayFromImage(all)
    maxarray = itk.GetArrayFromImage(max) 


else:
    print("loading vti...")
    all_path = os.path.join(inpath + allfilename + ".vti")
    max_path = os.path.join(inpath + maxfilename + ".vti")
    allarray = vtk_data_loader(all_path)
    maxarray = vtk_data_loader(max_path)

output = compound(allarray, maxarray)

if mhd:
    print("saving as mhd...")
    # array to image
    output = itk.GetImageFromArray(output)
    output.SetSpacing(spa)
    output.SetOrigin(ori)

    # save output file(.mhd)
    Out_Img = os.path.join(outpath + allfilename + "_comp.mhd")
    itk.imwrite(output, Out_Img)

else:
    print("saving as vti...")
    # array to image
    output = numpy_to_vtk(output)
    # save output file(.vti)
    Out_Img = os.path.join(outpath + allfilename + "comp.vti")
    save_vtk(output, Out_Img)

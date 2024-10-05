import itk
import vtk
import os
from vtkmodules.util import numpy_support
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
from skimage.filters import frangi
from utils.utils_for_frangi import triming, volume_processing
from utils.utils_for_frangi import norm_0_255
from utils.utils_for_frangi import binarizaiton

inpath = "D:\\takahashi_k\\frangi\\sigma_th"
outpath = "D:\\takahashi_k\\frangi\\sigma_th"
filename = "\\IM_0022-Origin"
mhd = False                 #mhd->True  vti->False
sigma = range(1, 10, 1)     #sigma of frangi
#th = range(10, 100, 10)     #th for binarize
th = 10
N = 15                      #trim this range edge voxel

if mhd:
    print("loading mhd...")
    In_Img = os.path.join(inpath + filename + ".mhd")
    #print(In_Img)
    #image to array
    input = itk.imread(In_Img)
    spa = input.GetSpacing()
    ori = input.GetOrigin()
    array = itk.GetArrayFromImage(input)


else:
    print("loading vti...")
    In_Img = os.path.join(inpath + filename + ".vti")
    array = vtk_data_loader(In_Img)

#normalization
array = norm_0_255(array)


for i in sigma:
    #for j in th:
    # frangi filter
    print(f"frangi filter by sigma={i}")
    output = frangi(array,sigmas=[i],alpha=0.5,beta=0.5,gamma=15,black_ridges=True)

    #normalization
    output = norm_0_255(output)

    #noise
    print("noise processing")
    output = triming(output, N)
    output = volume_processing(output)

    #binarizetion
    print("binarizaiton")
    output = binarizaiton(output, th)

    #noise
    # print("noise processing")
    # output = volume_processing(output)


    if mhd:
        print("saving as mhd...")
        # array to image
        output = itk.GetImageFromArray(output)
        output.SetSpacing(spa)
        output.SetOrigin(ori)

        # save output file(.mhd)
        Out_Img = os.path.join(outpath + filename + f"_sigma({i})_th({th}).mhd")
        itk.imwrite(output, Out_Img)

    else:
        print("saving as vti...")
        # array to image
        output = numpy_to_vtk(output)
        # save output file(.vti)
        Out_Img = os.path.join(outpath + filename + f"_sigma({i})_th({th})_nonVP.vti")
        save_vtk(output, Out_Img)

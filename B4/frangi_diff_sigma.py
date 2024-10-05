import itk
import vtk
from skimage.filters import frangi
import os
import numpy as np
from vtkmodules.util import numpy_support
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk

def frangi_max_all_sigma(array, sigma, Maxs):
    #creat 0 array
    # [x, y, z] = array.shape                         
    # print(x, y, z)
    # Allsigma = [[[0]*z]*y]*x
    Allsigma = np.zeros_like(array)

    #apply frangi filter
    for N in sigma:
        # sigmas = [N]
        # print(f"sigma = {sigmas}")
        output = frangi(array,sigmas=[N],alpha=0.5,beta=0.5,gamma=15,black_ridges=True)
    
        compare = output > Allsigma                             #output > Allsigma:1, output =< Allsigma:0
        inverse_comp = ~compare                                 #compareを反転(0->1, 1->0)
        Allsigma = compare*output + Allsigma*inverse_comp       #Allsigmaよりoutputが大きい輝度値はoutputに置き換え、それ以外はAllsigmaのまま

        if N==Maxs:
            Maxsigma = output
    
    return(Allsigma, Maxsigma)
    

inpath = "D:\\takahashi_k\\DataForAnalyse"
outpath = "D:\\takahashi_k\\frangi\\for_analyse"
filename = "\\IM_0022-Origin"
mhd = False                 #mhd->True  vti->False
sigma = range(1, 10, 1)

Maxs = max(sigma)
mins = min(sigma)

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

array = array*(255/array.max())                 #輝度値を0-255で正規化
print("frangi filter")
Allsigma, Maxsigma = frangi_max_all_sigma(array, sigma, Maxs)

if mhd:
    print("saving as mhd...")
    # array to image
    Allsigma = itk.GetImageFromArray(Allsigma)
    Allsigma.SetSpacing(spa)
    Allsigma.SetOrigin(ori)

    Maxsigma = itk.GetImageFromArray(Maxsigma)
    Maxsigma.SetSpacing(spa)
    Maxsigma.SetOrigin(ori)

    # save output file(.mhd)
    all_Out_path = os.path.join(outpath + filename + f"_frg({mins}-{Maxs}).mhd")
    max_Out_path = os.path.join(outpath + filename + f"_frg({Maxs}).mhd")
    itk.imwrite(Allsigma, all_Out_path)
    itk.imwrite(Maxsigma, max_Out_path)

else:
    print("saving as vti...")
    # array to image
    Allsigma = numpy_to_vtk(Allsigma)
    Maxsigma = numpy_to_vtk(Maxsigma)

    # save output file(.vti)
    all_Out_path = os.path.join(outpath + filename + f"_frg({mins}-{Maxs}).vti")
    max_Out_path = os.path.join(outpath + filename + f"_frg({Maxs}).vti")
    save_vtk(Allsigma, all_Out_path)
    save_vtk(Maxsigma, max_Out_path)

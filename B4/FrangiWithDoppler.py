import itk
import vtk
import os
import time
from vtkmodules.util import numpy_support
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
from utils.utils_for_frangi import frangi_max_all_sigma
from utils.utils_for_frangi import triming, volume_processing
from utils.utils_for_frangi import compound_with_doppler
from utils.utils_for_frangi import norm_0_255
from utils.utils_for_frangi import binarizaiton

inpath = "D:\\takahashi_k\\DataForAnalyse\\PATIENT4\\part2"
outpath = "D:\\takahashi_k\\frangi\\withdoppler\\patient4\\part2"
mhd = False                 #mhd->True  vti->False
dark = True               #dark vessel->True / bright vessel->Fals
sigma = range(1, 10, 1)     #sigma of frangi
N = 15                      #trim this range edge voxel
th = 10

def frangi_with_doppler(num):

    filename = f"\\IM_{num}-Origin"
    doppler = f"\\IM_{num}-doppler"
    print(f"start analyse {num} data")
    start_time = time.time()

    if mhd:
        print("loading mhd...")
        In_Img = os.path.join(inpath + filename + ".mhd")
        D_Img = os.path.join(inpath + doppler + ".mhd")
        #print(In_Img)
        #image to array
        input = itk.imread(In_Img)
        spa = input.GetSpacing()
        ori = input.GetOrigin()
        array = itk.GetArrayFromImage(input)

        input = itk.imread(D_Img)
        spa = input.GetSpacing()
        ori = input.GetOrigin()
        darray = itk.GetArrayFromImage(input)
        

    else:
        print("loading vti...")
        In_Img = os.path.join(inpath + filename + ".vti")
        D_Img = os.path.join(inpath + doppler + ".vti")
        array = vtk_data_loader(In_Img)
        darray = vtk_data_loader(D_Img)


    #normalization
    array = norm_0_255(array)

    # frangi filter
    print("frangi filter")
    maxs = max(sigma)
    Allsigma, Maxsigma = frangi_max_all_sigma(array, sigma, maxs, dark)

    #normalization
    Allsigma = norm_0_255(Allsigma)
    Maxsigma = norm_0_255(Maxsigma)

    #noise
    print("noise processing")
    Allsigma = triming(Allsigma, N)
    Allsigma = volume_processing(Allsigma)
    Maxsigma = triming(Maxsigma, N)
    Maxsigma = volume_processing(Maxsigma)

    #compound all sigma img to max sigma img
    print("compound all to max")
    output = compound_with_doppler(Allsigma, Maxsigma, darray)


    if mhd:
        print("saving as mhd...")
        # array to image
        output = itk.GetImageFromArray(output)
        output.SetSpacing(spa)
        output.SetOrigin(ori)

        # save output file(.mhd)
        Out_Img = os.path.join(outpath + filename + "_frgD.mhd")
        itk.imwrite(output, Out_Img)

    else:
        print("saving as vti...")
        # array to image
        output = numpy_to_vtk(output)
        # save output file(.vti)
        Out_Img = os.path.join(outpath + filename + "frgD.vti")
        save_vtk(output, Out_Img)

if __name__ == '__main__':
    for i in range(3, 41, 2):
        if i < 10:
            num = "000" + f"{i}"
            frangi_with_doppler(num)

        elif i < 100:
            num = "00" + f"{i}"
            frangi_with_doppler(num)

        else:
            num = "0" + f"{i}"
            frangi_with_doppler(num)
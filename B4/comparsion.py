import numpy as np
import itk
import vtk
from vtkmodules.util import numpy_support
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk

In_Img1 = "D:/takahashi_k/simulation/add_noise/vessel/noise_frg(1-5)coneected_vp(100).vti"
In_Img2 = "D:/takahashi_k/simulation/add_noise/vessel/noise_frg(1-5)coneected_vp(4618byOTSU).vti"
Out_Img = "D:/takahashi_k/simulation/add_noise/vessel/vp(100)-vp(4618byOTSU).vti"
mhd = False          #True->mhd False->vti

def diff(array1, array2):
    #take the difference
    diff = array1-array2 #輝度値の差
    output = np.abs(diff)   #輝度値の差の絶対値
    #output = output*(255/output.max())
    return(output)
    

if mhd:
    #image to array
    input1 = itk.imread(In_Img1)
    array1 = itk.GetArrayFromImage(input1)
    spa = input1.GetSpacing()
    ori = input1.GetOrigin()

    input2 = itk.imread(In_Img2)
    array2 = itk.GetArrayFromImage(input2)

    output = diff(array1, array2)
    # array to image
    output = itk.GetImageFromArray(output)
    output.SetSpacing(spa)
    output.SetOrigin(ori)

    # save output file(.mhd)
    itk.imwrite(output, Out_Img)


else:
    array1, spa, ori = vtk_data_loader(In_Img1)
    array2, spa, ori = vtk_data_loader(In_Img2)
    output = diff(array1, array2)
    output = numpy_to_vtk(output, spa, ori)
    save_vtk(output, Out_Img)




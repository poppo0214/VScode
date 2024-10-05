import numpy as np
import itk
from skimage.filters import frangi

In_Img = "D:\\takahashi_k\\frangi\\sigma_th\\IM_0022-Origin.mhd"
Out_Img =  "D:\\takahashi_k\\frangi\\sigma_th\\IM_0022-Origin_frg7_norm(0-255).mhd"

#image to array
input = itk.imread(In_Img)
spa = input.GetSpacing()
ori = input.GetOrigin()
array = itk.GetArrayFromImage(input)
array = array*(255/array.max())

# frangi filter
output = frangi(array,sigmas=[7],alpha=0.5,beta=0.5,gamma=15,black_ridges=True)
output = output*(255/output.max())

# array to image
output = itk.GetImageFromArray(output)
output.SetSpacing(spa)
output.SetOrigin(ori)

# save output file(.mhd)
itk.imwrite(output, Out_Img)

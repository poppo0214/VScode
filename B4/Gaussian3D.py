import numpy as np
import itk
from skimage.filters import gaussian

In_Img = "D:/takahashi_k/frangi/origin.mhd"
Out_Img = "D:/takahashi_k/frangi/gs/origin_gs1.mhd"

#image to array
input = itk.imread(In_Img)
spa = input.GetSpacing()
ori = input.GetOrigin()
array = itk.GetArrayFromImage(input)

# gaussian filter
output =  gaussian(array, sigma=1)
output = output*(255/output.max())

# array to image
output = itk.GetImageFromArray(output)

output.SetSpacing(spa)

output.SetOrigin(ori)

# save output file(.mhd)
itk.imwrite(output, Out_Img)

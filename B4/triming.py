import numpy as np
import itk
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk

def triming(output, N):
    dim = output.shape
    x = dim[0]
    y = dim[1]
    z = dim[2]

    output[:N, :, :] = 0
    output[:, :N, :] = 0
    output[:, :, :N] = 0

    output[x-N:, :, :] = 0
    output[:, y-N:, :] = 0
    output[:, :, z-N:] = 0

    return output

In_Img = "D:/takahashi_k/frangi/normalization/origin_frg9_norm(0-255).mhd"
Out_Img = "D:/takahashi_k/frangi/triming/origin_frg9_norm(0-255)_trm20.mhd"
N = 15         #周り何ボクセルを取り除くか

#image to array
input = itk.imread(In_Img)
spa = input.GetSpacing()
ori = input.GetOrigin()
array = itk.GetArrayFromImage(input)

#triming
#array = np.random.rand(10, 15, 20)
# array = vtk_data_loader(In_Img)
output = triming(array, N)

# array to image
output = itk.GetImageFromArray(output)
output.SetSpacing(spa)
output.SetOrigin(ori)

# save output file(.mhd)
# output = numpy_to_vtk(output)
# save_vtk(output, Out_Img)
itk.imwrite(output, Out_Img)
# output = np.array(output)
# output = output.flatten()
#np.savetxt("D:/takahashi_k/frangi/triming/origin_frg9_norm(0-255)_trm.txt", output)
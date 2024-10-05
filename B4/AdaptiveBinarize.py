import itk
import cv2
import numpy as np

In_Img = "D:/takahashi_k/frangi/compound/origin_frg(1-9)_trm15_comp.mhd"
Out_Img = "D:/takahashi_k/frangi/binarized/origin_frg(1-9)_trm15_Adp(mean31).mhd"

#image to array
input = itk.imread(In_Img)
spa = input.GetSpacing()
ori = input.GetOrigin()
array = itk.GetArrayFromImage(input)

# adaptive binarize
dim = array.shape
x = dim[0]
y = dim[1]
z = dim[2]

rearray = np.clip(array, 0, 255).astype(np.uint8)
rearray = np.reshape(rearray, (x, y*z))
print(rearray.dtype, rearray.shape)

output = cv2.adaptiveThreshold(rearray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 0)
output = np.reshape(output, (x, y, z))

# array to image
output = itk.GetImageFromArray(output)
output.SetSpacing(spa)
output.SetOrigin(ori)

# save output file(.mhd)
itk.imwrite(output, Out_Img)

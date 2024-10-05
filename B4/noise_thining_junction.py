from skimage.morphology import skeletonize_3d
import numpy as np
import itk
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
from utils.labeling import labeling
import time
import numpy as np

inpath = "D:\\takahashi_k\\frangi\\thining_junction"
outpath = "D:\\takahashi_k\\frangi\\thining_junction"
filename = f"\\model"
mhd = False                 #mhd->True  vti->False

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
    

labeledarray, N = labeling(array)

for num in range(1, N+1):
    extracted_image = (labeledarray==num)    
    centerline = skeletonize_3d(extracted_image)
    (x, y, z) = extracted_image.shape

    vessel = False
    for i in range(1, z, 1):
        for j in range(1, y, 1):
            for k in range(1, x, 1):
                count = 0
                for a in range(-1, 2, 1):
                    for b in range(-1, 2, 1):
                        for c in range(-1, 2, 1):
                            count += centerline[k+c, j+b, i+a]
                if count >= 4:
                    vessel = True
    if vessel:
        temp = np.where(labeledarray == num , 1, 0)
        output = output + temp

output = output * array




if mhd:
    print("saving as mhd...")
    # array to image
    output = itk.GetImageFromArray(output)
    output.SetSpacing(spa)
    output.SetOrigin(ori)

    # save output file(.mhd)
    Out_Img = os.path.join(outpath + filename + "_frg.mhd")
    itk.imwrite(output, Out_Img)

else:
    print("saving as vti...")
    # array to image
    output = numpy_to_vtk(output)
    # save output file(.vti)
    Out_Img = os.path.join(outpath + filename + "_frg.vti")
    save_vtk(output, Out_Img)

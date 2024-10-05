import numpy as np
import itk
import cv2
import vtk
from vtkmodules.util import numpy_support
from utils.labeling import labeling
from scipy import ndimage
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk

inpath = "D:\\takahashi_k\\simulation\\add_noise\\vessel\\"
outpath = inpath
filename = "noise_close(5)_mask(5)"


def volume_processing(array):
    print("volume processing")
    labeledarray, N = labeling(array)

    areas = []
    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        areas.append(count)

    # areas = np.array(areas)
    # areas = np.uint16(areas)                            
    maxarea = max(areas)
    print(maxarea)
    output = np.zeros_like(array)

    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        #print(count)
        if count == maxarea:
            output = np.where(labeledarray == num , 1, 0)

    output = output * array
    return(output, maxarea)

# def volume_processing_rev(array):
#     print("volume processing(rev)")
#     labeledarray, N = labeling(array)

#     output = np.zeros_like(array)

#     areas = ndimage.sum(array, labeledarray, range(N + 1))
#     maxarea = max(areas)
#     mask_area = areas == maxarea                               
#     output = mask_area * array

#     return(output, maxarea)

if __name__ == '__main__':

    print("loading vti...")
    In_Img = os.path.join(inpath + filename + ".vti")
    array, spa, ori = vtk_data_loader(In_Img)

    
    #volume_processing
    output, __ = volume_processing(array)
    #output, __ = volume_processing_rev(array)

    print("saving as vti...")
    # array to image
    output = numpy_to_vtk(output, spa, ori)
    # save output file(.vti)
    Out_Img = os.path.join(outpath + filename + f"_vp(biggest).vti")
    save_vtk(output, Out_Img)


import os
import numpy as np
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import glob
import subprocess as sp
from scipy.ndimage import label
from scipy import ndimage
import re
import itk
txtroot = ""

def labeling(array):
    #labeling
    str_3D = np.array([[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                        [[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]],
                        [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]], dtype='uint8')
    labeledarray, N = label(array, structure = str_3D)

    return(labeledarray, N)


def volume_processing_rev(array, th = None):
    print("volume processing(rev)")
    labeledarray, N = labeling(array)

    output = np.zeros_like(array)

    areas = ndimage.sum(array, labeledarray, range(N + 1))
    mask_area = areas < th                                 
    remove_voxel = mask_area[labeledarray]
    labeledarray[remove_voxel] = 0
    output = (labeledarray > 0).astype(np.int8)
    output = output * array
    return(output)

if __name__ == '__main__':
    f = open(txtroot, "r")
    while True:
        data = f.readline()
        [root, th] = data.split()
        th = int(th)
        #input vti
        print("loading vti data...")
        array, spa, ori = vtk_data_loader(root)
        output = volume_processing_rev(array)
        
        #save as vti
        print("saving as vti...")
        output = numpy_to_vtk(output, spa, ori)
        outroot = re.sub('binarized', 'volumeProcessed', root)
        outroot = re.sub('.vti', '', outroot)
        outroot = os.path.join(outroot + f"_vp.vti")
        print(outroot)
        save_vtk(output, outroot)
        if data == '':
            break


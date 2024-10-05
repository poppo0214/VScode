import vtk
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import numpy as np
from scipy.ndimage import label

In_path = 'd:\\takahashi_k\\simulation\\vessel\\v3\\'
Out_path = In_path
filename = "vessel_v3_rev_pad"


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

if __name__ == '__main__':
    In_root = os.path.join(In_path + filename + '.vti')
    Out_root = os.path.join(Out_path + filename + f'_label.vti')
    print(In_root)

    print("loading vti file...")
    array, spa, ori = vtk_data_loader(In_root)

    print("labeling...")
    output, N = label(array)

    #save as vti
    print("saving as vti...")
    output = numpy_to_vtk(output, spa, ori)
    save_vtk(output, Out_root)    
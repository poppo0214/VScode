from scipy import ndimage
from scipy.ndimage import label
import numpy as np
import itk


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

def labeling_with_volume_processing(array, th):
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
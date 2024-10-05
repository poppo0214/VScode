import vtk
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
from vtkmodules.util import numpy_support
import numpy as np
from skimage.filters import frangi
from scipy.ndimage import label
import cv2
from scipy import ndimage
import glob

In_img = 'd:\\takahashi_k\\\simulation\\vessel\\v3\\'
Out_img = In_img
filename = 'vessel_v3_rev'
In_root = os.path.join(In_img + filename + '.vti')
Out_root = os.path.join(Out_img + filename + '_pad.vti')

print("loading vti file...")
array, spa, ori = vtk_data_loader(In_root)

print("padding...")
# output = np.pad(array, [(前0画像数, 後0画像数), (上0埋め数, 下0埋め数), (左0埋め数, 右0埋め数)], 'constant')
output = np.pad(array, [(40, 40), (40, 40), (40, 40)], 'constant')

#save as vti
print("saving as vti...")
output = numpy_to_vtk(output, spa, ori)
save_vtk(output, Out_root)


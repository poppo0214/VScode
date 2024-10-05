import open3d as o3d
import numpy as np
import copy
import os
import glob
import time
import os
import sys
import copy
import numpy as np
import open3d as o3d
import glob
import vtk
from vtkmodules.util import numpy_support
from skimage.morphology import opening, closing, cube
from utils.utils_vtk import numpy_to_vtk, save_vtk
import itertools
import joblib as jl
import multiprocessing as mp
from multiprocessing import shared_memory





list1 = [[1,2,3], [4,5,6], [7,8,9]]
array1 = np.asarray(list1)
list2 = [[10,11,12], [13,14,15], [16,17,18]]
array2 = np.asarray(list2)

print(array1.shape)
print(array2.shape)
print(array1 * array2)
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import numpy as np
from skimage.morphology import opening, closing, cube
import cv2
import glob

def process(array, kernel):
    #closing(膨張→収縮)
    # output = closing(array > 0, square(kernel))
    output = closing(array, cube(kernel))

    #opening(収縮→膨張)
    #output = opening(array)
    
    return(output)


if __name__ == '__main__':
    In_path = r'D:\takahashi_k\registration(model)\forUSE\original\filled'
    Out_path = r'D:\takahashi_k\registration(model)\forUSE\original\filled'
   
    pad_sz = 20

    rootlist = glob.glob(f'{In_path}/*rev(filled).vti')
    for root in rootlist:
        print(root)
        array, spa, ori = vtk_data_loader(root)
        output = np.pad(array, [(pad_sz, pad_sz), (pad_sz, pad_sz), (pad_sz, pad_sz)], 'constant')

        #save as vti
        print("saving as vti...")
        filename = root.replace(In_path, "")
        filename = filename.replace(".vti", "")
        Out_root = os.path.join(Out_path  + f"{filename}_padding({pad_sz}).vti")
        print(Out_root)
        output = numpy_to_vtk(output, spa, ori)
        save_vtk(output, Out_root)


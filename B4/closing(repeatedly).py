from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import numpy as np
from skimage.morphology import dilation, erosion, cube

In_path = 'd:\\takahashi_k\\simulation\\closing\\'
Out_path = In_path
filename = "gauss(non)_frg(1,2,1)"
num = 5                          #何回処理を繰り返すか

def process(array):
    output = array
    #dilation(膨張)
    for i in range(0, num, 1):
        output = dilation(output, cube(1))

    #erosion(収縮)
    for j in range(0, num, 1):
        output = erosion(output, cube(1))
    
    return(output)


if __name__ == '__main__':
    In_root = os.path.join(In_path + filename + '.vti')
    Out_root = os.path.join(Out_path + filename + f'_close_rep({num}).vti')
    #Out_root = os.path.join(Out_path + filename + f'_close(default).vti')
    print(In_root)

    print("loading vti file...")
    array, spa, ori = vtk_data_loader(In_root)

    print("opening_closeing")
    output = process(array)

    #output = frangi(array, sigmas=sgms, black_ridges=False)
    output = output * (255/np.max(output))

    #save as vti
    print("saving as vti...")
    output = numpy_to_vtk(output, spa, ori)
    save_vtk(output, Out_root)
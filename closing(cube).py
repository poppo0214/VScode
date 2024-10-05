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
    #kernel = 5                         #クロージング処理のカーネルサイズ

    kernel_list = [10]
    # for kernel in kernel_list:
    #     os.makedirs(os.path.join((Out_path) + f"_closing({kernel})"), exist_ok=True)
    rootlist = glob.glob(f'{In_path}/*Y-deform-rev(filled)_padding(20).vti')
    for root in rootlist:
        print(root)
        array, spa, ori = vtk_data_loader(root)

        for kernel in kernel_list:
            print("closeing")
            output = process(array, kernel)

            #output = frangi(array, sigmas=sgms, black_ridges=False)
            #output = output * (255/np.max(output))

            #save as vti
            print("saving as vti...")
            filename = root.replace(In_path, "")
            filename = filename.replace(".vti", "")
            Out_root = os.path.join(Out_path  + f"{filename}_closing({kernel}).vti")
            print(Out_root)
            output = numpy_to_vtk(output, spa, ori)
            save_vtk(output, Out_root)


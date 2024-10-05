from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import numpy as np
from skimage.filters import gaussian

In_path = 'd:\\takahashi_k\\temporary space\\'
filename = "Shimizu_IM_0014-Origin_sigmoid_ShuicaiWu(5)"
sigmas = range(1, 11, 1)
gaussis = range(1, 11,1)

def process(array, sgm=1):
    #inversion
    #output = np.where(array==1, 0, 1)

    #normalization
    output = array * (255/np.max(array))

    #gauusian filter
    output = gaussian(output, sigma=sgm)

    #normalization
    output = output * (255/np.max(output))
    
    return(output)


if __name__ == '__main__':
    #gaussでfor文を回す
    # for gauss in gaussis:
    #     print(gauss)
    #     path = os.path.join(In_path + f"gauss({gauss})\\gauss({gauss})")
    #     In_root = os.path.join(path + ".vti")
    #     print("loading vti file...")
    #     array, spa, ori = vtk_data_loader(In_root)

    #     output = process(array)

    #     #save as vti
    #     print("saving as vti...")
    #     Out_root = os.path.join(path + '_norm.vti')    
    #     output = numpy_to_vtk(output, spa, ori)
    #     save_vtk(output, Out_root)    
    
    #sigmaでfor文を回す
    In_root = os.path.join(In_path + filename + '.vti')
    print("loading vti file...")
    array, spa, ori = vtk_data_loader(In_root)

    for i in sigmas:
        output = process(array, sgm=i)
        outname = f'gauss({i})'

        #save as vti
        print("saving as vti...")
        Out_path = In_path + outname
        os.mkdir(Out_path)
        
        Out_root = os.path.join(Out_path + "\\" + outname + '.vti')    
        output = numpy_to_vtk(output, spa, ori)
        save_vtk(output, Out_root)
import numpy as np
from utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import numpy as np
import glob

In_path = 'd:\\takahashi_k\\database\\us\\kobayashi\\origin'
#gaussis = range(1, 11, 1)
Out_path = In_path
In_name = "0062-Origin"
#Out_name = "Origin_byYukino(2)_frg(1)"


if __name__ == '__main__':
    rootlist = glob.glob(f"{In_path}/*.vti")

    for root in rootlist:
        print(root)
        print("loading vti file...")
        array, spa, ori = vtk_data_loader(root)
        #transpose
        # array = np.transpose(array, (2, 1, 0))
        #change spacing
        spa = (spa[2], spa[1], spa[0])

        #save as vti
        print("saving as vti...")
        filename = root.replace(In_path, "")
        filename = filename.replace(".vti", "")
        Out_root = os.path.join(Out_path + '_change_spacing' + filename + '.vti')
        print(Out_root)
        output = numpy_to_vtk(array, spa, ori)
        save_vtk(output, Out_root)
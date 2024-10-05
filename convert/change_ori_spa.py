import os
import vtk
from vtkmodules.util import numpy_support
import numpy as np
import glob
from utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import pydicom

# if __name__ == '__main__':
#     In_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Annotation\VTI"
#     Out_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Annotation\VTI_rev"
    
#     rootlist = glob.glob(f'{In_path}/*.vti')

#     for root in rootlist:
#         print(root)
#         array, spa, ori = vtk_data_loader(root)
#         ori = [255.50000000000003, -28.405355603592582, 116.5]
#         spa = [0.36813678694904567, 0.3157286665464158, 0.5453666041673727]
#         filename = root.replace(In_path, "")
#         filename = filename.replace("IM_", "")
#         #filename = filename.replace("_seg", "-seg")
#         Out_root = os.path.join(Out_path + filename)
#         output = numpy_to_vtk(array, spa, ori)
#         save_vtk(output, Out_root)

if __name__ == '__main__':
    In_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Annotation\VTI(0,0,0)"
    reference_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\DICOM"
    Out_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Annotation\VTI(filled)"
    
    rootlist = glob.glob(f'{In_path}/*.vti')

    for root in rootlist:
        print(root)
        reference_root = os.path.join(reference_path + "\\" )
        array, __, __ = vtk_data_loader(reference_root)
        data = pydicom.dcmread(root)
        spacing = data[0x200d, 0x3303].value
        spacing = str(spacing)
        spacing = spacing.lstrip("['")
        spacing = spacing.rstrip("']")
        spacing = spacing.split("', '")
        for i in range(len(spacing)):
            spacing[i] = float(spacing[i])
        origin = data[0x200d, 0x3304].value
        origin = str(origin)
        origin = origin.lstrip("['")
        origin = origin.rstrip("']")
        origin = origin.split("', '")
        for i in range(len(origin)):
            origin[i] = float(origin[i])


        filename = root.replace(In_path, "")
        filename = filename.replace("IM_", "")
        #filename = filename.replace("_seg", "-seg")
        Out_root = os.path.join(Out_path + filename)
        output = numpy_to_vtk(array, spacing, origin)
        save_vtk(output, Out_root)
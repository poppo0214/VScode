import itk
import vtk
from vtkmodules.util import numpy_support
import os
import glob

if __name__ == '__main__':
    # dcm_path = "D:\\takahashi_k\\dicom\\patient2"
    # filename = "\\IM_0002"
    # out_path = "D:\\takahashi_k\\testDICOM\\patient2"
    # In_Dcm = os.path.join(dcm_path + filename)
    # Out_origin = os.path.join(out_path + filename + "-origin.vti")
    #Out_doppler = os.path.join(out_path + filename + "-doppler.vti")

    in_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\VTI_rev"
    rootlist = glob.glob(f"{in_path}/*.vti")
    out_path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI\Origin\MHD_rev"

    for in_root in rootlist:
        filename = in_root.replace(in_path, "")
        filename = filename.replace(".vti", "")
        Out_root = os.path.join(out_path + filename + ".mhd")

        #image to array
        vtk_reader = vtk.vtkXMLImageDataReader()
        vtk_reader.SetFileName(in_root)
        vtk_reader.Update()
        data = vtk_reader.GetOutput()
        dims = data.GetDimensions()
        spa = data.GetSpacing()
        ori = data.GetOrigin()

        output = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
        output= output.reshape(dims[2], dims[1], dims[0])

        output = itk.GetImageFromArray(output)
        output.SetSpacing(spa)
        output.SetOrigin(ori)

        # save output file(.mhd)
        itk.imwrite(output, Out_root)

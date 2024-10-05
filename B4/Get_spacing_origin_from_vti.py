import os
import vtk
from vtkmodules.util import numpy_support
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk

inpath = "D:\\takahashi_k\\frangi\\test"
filename = "\\patient1-Origin"

print("loading vti...")
In_Img = os.path.join(inpath + filename + ".vti")
# array = vtk_data_loader(In_Img)

vtk_reader = vtk.vtkXMLImageDataReader()
vtk_reader.SetFileName(In_Img)
vtk_reader.Update()
vtk_data = vtk_reader.GetOutput()

spacing = vtk_data.GetSpacing()
origin = vtk_data.GetOrigin()
print("spacing: ", type(spacing), spacing, "\norigin: ", type(origin), origin)




import pydicom
import os
import itk
import vtk
from vtkmodules.util import numpy_support
import numpy as np

inpath = "d:/takahashi_k/originData/byYukino/ito/62"
outpath = inpath
filename = "\\IM_0062-Origin"

def numpy_to_vtk(data, spa, ori, multi_component=False, type='char'):
  '''
  multi_components: rgb has 3 components
  type:float or char
  '''
  if type == 'float':
    data_type = vtk.VTK_FLOAT
  elif type == 'char':
    data_type = vtk.VTK_UNSIGNED_CHAR
  else:
    raise RuntimeError('unknown type')
  if multi_component == False:
    if len(data.shape) == 2:
      data = data[:, :, np.newaxis]
    flat_data_array = data.transpose(2,1,0).flatten()
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    shape = data.shape
  else:
    assert len(data.shape) == 3, 'only test for 2D RGB'
    flat_data_array = data.transpose(1, 0, 2)
    flat_data_array = np.reshape(flat_data_array, newshape=[-1, data.shape[2]])
    vtk_data = numpy_support.numpy_to_vtk(num_array=flat_data_array, deep=True, array_type=data_type)
    shape = [data.shape[0], data.shape[1], 1]
  img = vtk.vtkImageData()
  img.GetPointData().SetScalars(vtk_data)
  img.SetDimensions(shape[0], shape[1], shape[2])
  img.SetSpacing(spa[0], spa[1], spa[2])
  img.SetOrigin(ori[0], ori[1], ori[2])
  return img

def save_vtk(img, output_path):
  writer = vtk.vtkXMLImageDataWriter()
  writer.SetFileName(output_path)
  writer.SetInputData(img)
  writer.Write()

if __name__ == '__main__':
    In_Img = os.path.join(inpath + filename + ".mhd")
    Out_Img = os.path.join(outpath + filename + ".vti")

    print("reading mhd...")
    input = itk.imread(In_Img)
    spa = input.GetSpacing()
    ori = input.GetOrigin()
    array = itk.GetArrayFromImage(input)

    print("saving as vti...")
    # save output file(.vti)
    output = numpy_to_vtk(array, spa, ori)
    # save output file(.vti)
    Out_Img = os.path.join(outpath + filename + ".vti")
    save_vtk(output, Out_Img)

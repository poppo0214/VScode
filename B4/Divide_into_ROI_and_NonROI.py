import os
import itk
from utils.utils_vtk import vtk_data_loader, save_vtk
import vtk
from vtkmodules.util import numpy_support
import numpy as np

inpath1 =   "D:\\takahashi_k\\frangi\\test"             #segmentationed
filename1 = "\\patient3-Origin_gt"
inpath2 =   "D:\\takahashi_k\\frangi\\test"             #original or frangi
filename2 = "\\patient3-Origin_frg"
outpath =   "D:\\takahashi_k\\frangi\\test"   
mhd = False

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

if __name__ == '__main__':
    if mhd:
        print("loading mhd...")
        In_Img1 = os.path.join(inpath1 + filename1 + ".mhd")
        In_Img2 = os.path.join(inpath2 + filename2 + ".mhd")

        #image to array
        input1 = itk.imread(In_Img1)
        array1 = itk.GetArrayFromImage(input1)
        input2 = itk.imread(In_Img2)
        array2 = itk.GetArrayFromImage(input2)

        spa = input1.GetSpacing()
        ori = input1.GetOrigin()


    else:
        print("loading vti...")
        In_Img1 = os.path.join(inpath1 + filename1 + ".vti")
        array1 = vtk_data_loader(In_Img1)
        In_Img2 = os.path.join(inpath2 + filename2 + ".vti")
        array2 = vtk_data_loader(In_Img2)

        vtk_reader = vtk.vtkXMLImageDataReader()
        vtk_reader.SetFileName(In_Img1)
        vtk_reader.Update()
        vtk_data = vtk_reader.GetOutput()

        spa = vtk_data.GetSpacing()
        ori = vtk_data.GetOrigin()

    array2 = array2*(1/array2.max())

    #multiplication(関心領域[ROI]は輝度値そのまま、非関心領域[nonROI]は輝度値０)
    ROI = array1*array2 #輝度値の積
    nonROI = array2 - ROI
    ROI = ROI*(255/ROI.max())
    nonROI = nonROI*(255/nonROI.max())


    if mhd:
    # array to image
        Out_Img1 = os.path.join(outpath + filename2 + "_ROI.mhd")                      #血管領域
        Out_Img2 = os.path.join(outpath + filename2 +  "_nonROI.mhd" )                   #非血管領域（＝ノイズ）
        output1 = itk.GetImageFromArray(ROI)
        output1.SetSpacing(spa)
        output1.SetOrigin(ori)
        output2 = itk.GetImageFromArray(nonROI)
        output2.SetSpacing(spa)
        output2.SetOrigin(ori)

        # save output file(.mhd)
        itk.imwrite(output1, Out_Img1)
        itk.imwrite(output2, Out_Img2)

    else:
        Out_Img1 = os.path.join(outpath + filename2 + "_ROI(frg).vti" )                      #血管領域
        Out_Img2 = os.path.join(outpath + filename2 +  "_nonROI(frg).vti" )                   #非血管領域（＝ノイズ）
        output1 = numpy_to_vtk(ROI, spa, ori)
        output2 = numpy_to_vtk(nonROI, spa, ori)
        save_vtk(output1, Out_Img1)
        save_vtk(output2, Out_Img2)
import vtk
from vtkmodules.util  import numpy_support
import os
import numpy as np
import pandas as pd
import glob
from scipy.ndimage import morphology
import warnings
warnings.simplefilter('ignore')
import sys
sys.path.append(os.pardir)
from utils.utils_vtk import vtk_to_numpy, save_vtk, numpy_to_vtk

def vtk_data_loader_rev(data_path):
  """
  This function is to load vtk data
  Args
      data_path: vtk data path
  Return: vtk data transformed to numpy
  """
  vtk_reader = vtk.vtkXMLImageDataReader()
  vtk_reader.SetFileName(data_path)
  vtk_reader.Update()
  vtk_data = vtk_reader.GetOutput()
  
  spa = vtk_data.GetSpacing()
  ori = vtk_data.GetOrigin()
  bounds = vtk_data.GetBounds()

  npdata = vtk_to_numpy(vtk_data).astype(np.float32)
  
  # data = np.zeros((x, y, z))

  return (npdata, spa, ori, bounds)

def padding(array1, array2, bounds1, bounds2, spa):
    diff_bounds = np.abs(np.asarray(bounds2) - np.asarray(bounds1))
    #print(diff_bounds)
    add_voxel = diff_bounds/[spa[0], spa[0], spa[1], spa[1], spa[2], spa[2]]
    array2 = np.pad(array2, [(int(add_voxel[0]), int(add_voxel[1])), 
                            (int(add_voxel[2]), int(add_voxel[3])),
                             (int(add_voxel[4]), int(add_voxel[5]))], "constant")
    
    #↑はintで丸めてるので、足りない可能性
    if int(array1.shape[0] - array2.shape[0]) != 0:
        array2 = np.pad(array2, [(int(array1.shape[0] - array2.shape[0]),0), (0,0), (0,0)], "constant")
    
    if int(array1.shape[1] - array2.shape[1]):
        array2 = np.pad(array2, [(0,0), (int(array1.shape[1] - array2.shape[1]),0), (0,0)], "constant")
    
    if int(array1.shape[2] - array2.shape[2]):
        array2 = np.pad(array2, [(0,0), (0,0), (int(array1.shape[2] - array2.shape[2]),0)], "constant")
    
    #print(array1.shape, array2.shape)
    return array2
 
 
def surfd(input1, input2, sampling, connectivity=1): 
    input_1 = np.atleast_3d(input1.astype(bool))
    input_2 = np.atleast_3d(input2.astype(bool))
    print(input_1.shape)
    print(input_2.shape)
 
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)
 
    S = input_1 ^ morphology.binary_erosion(input_1, conn)
    Sprime = input_2 ^ morphology.binary_erosion(input_2, conn) 
 
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
 
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])]) 
    asdd = sds.mean()
 
    return asdd

 
 
if __name__ == '__main__':
    """vtiデータを使用してください"""
    correct_path = r"D:\takahashi_k\database\us\divide" 
    #correct_name = r"IM_0328-Annotation"
    expand_path = r"D:\takahashi_k\database\us\divide\divide(1_8)_move(1-30)\icp\expand"
    csvroot = os.path.join(expand_path + "\\ASD.tsv")
    fcsv = open(csvroot, "a")
    fcsv.write("correct(before registrarion)\tsource(registrarioned)\ttarget\titration num\tASD\n")

    # correct_name = 
    # correct_root = os.path.join(correct_path + "\\" + correct_name + ".vti")
    # correct_array, spa1, ori1, bounds1 = vtk_data_loader_rev(correct_root)
    # correct_name = correct_root.replace(correct_path, "")
    # correct_name = correct_name.replace(".vti", "")
    # correct_name = correct_name.replace("\\", "")

    rootlist = glob.glob(f'{expand_path}/*.vti')
    for expand_root in rootlist:
        #vtiデータ読み込み
        expand_array, spa2, ori2, bounds2 = vtk_data_loader_rev(expand_root)
        expand_name = expand_root.replace(expand_path, "")
        name = expand_name.split("_")[0]
        num = expand_name.split("_")[2]
        correct_name = name + "_IM_" + num
        correct_root = os.path.join(correct_path + "\\" + correct_name + ".vti")
        correct_array, spa1, ori1, bounds1 = vtk_data_loader_rev(correct_root)
        correct_name = correct_root.replace(correct_path, "")
        correct_name = correct_name.replace(".vti", "")
        correct_name = correct_name.replace("\\", "")
        if correct_array.shape != expand_array.shape:
            expand_array = padding(correct_array, expand_array, bounds1, bounds2, spa2)
            rev_path = expand_root.replace(".vti", "")
            rev_root = os.path.join(rev_path + "_rev.vti")
            rev_vti = numpy_to_vtk(expand_array, spa1, ori1)
            save_vtk(rev_vti, rev_root)

        expand_name = expand_root.replace(expand_path, "")
        """名前取得（icp時の設定）"""
        name = expand_name.split("_icped_to_")
        source_name = str(name[0])
        target_number = name[1].replace(".vti", "")
        temp = target_number
        temp_list = temp.split("_")
        iteration_num = temp_list[-2]
        target_name = target_number.replace("_expand", "")
        target_name = target_name.replace(iteration_num, "")
        print(source_name, target_name, iteration_num)

        ASD = surfd(expand_array, correct_array, list(spa1), 1)
        print(ASD)
        fcsv.write(f"{correct_name}\t{source_name}\t{target_name}\t{iteration_num}\t{ASD}\n")
    fcsv.close()



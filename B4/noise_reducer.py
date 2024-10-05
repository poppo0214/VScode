import numpy as np
import itk
import cv2
import vtk
from vtkmodules.util import numpy_support
from utils.labeling import labeling
from scipy import ndimage
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import glob




def triming(array, N):
    dim = array.shape
    x = dim[0]
    y = dim[1]
    z = dim[2]
    #ばいとおつ
    output = np.array(array)
    
    output[:N, :, :] = 0
    output[:, :N, :] = 0
    output[:, :, :N] = 0
    
    output[x-N:, :, :] = 0
    output[:, y-N:, :] = 0
    output[:, :, z-N:] = 0
    

    return(output)


def mask_processing(mask_Img, array):
    vtk_reader = vtk.vtkXMLImageDataReader()
    vtk_reader.SetFileName(mask_Img)
    vtk_reader.Update()
    data = vtk_reader.GetOutput()
    dims = data.GetDimensions()
    mask = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
    mask = mask.reshape(dims[2], dims[1], dims[0])
    output = array * mask

    return(output)


def volume_processing(array, th = None):
    print(f"volume processing({th})")
    labeledarray, N = labeling(array)

    areas = []
    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        areas.append(count)

    areas = np.array(areas)
    areas = np.uint16(areas)

    if th == None:                      #大津の2値化による閾値の設定
        th, __ =  cv2.threshold(areas, 0, 1, cv2.THRESH_OTSU)
        print(f"th by OTSU = {th}")                            

    output = np.zeros_like(array)

    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        if count >= th:
            temp = np.where(labeledarray == num , 1, 0)
            output = output + temp

    output = output * array
    return(output, th)

def volume_processing_rev(array, th = None):
    print("volume processing(rev)")
    labeledarray, N = labeling(array)

    if th == None:                      #大津の2値化による閾値の設定
        th, __ =  cv2.threshold(areas, 0, 1, cv2.THRESH_OTSU)
        print(f"th by OTSU = {th}")

    output = np.zeros_like(array)

    areas = ndimage.sum(array, labeledarray, range(N + 1))
    mask_area = areas < th                                 
    remove_voxel = mask_area[labeledarray]
    labeledarray[remove_voxel] = 0
    output = (labeledarray > 0).astype(np.int8)
    output = output * array

    return(output, th)

def binarize(array, th):
    compare = array > th
    output = compare.astype(np.int8)
    return(output)

if __name__ == '__main__':
    inpath = r'd:\takahashi_k\PaperData\frg(1,17,2)_VP\frg(1,17,2)_bn(20)'
    outpath = r'd:\takahashi_k\PaperData\frg(1,17,2)_VP'
    bthlist = [10, 20, 30]
    vthlist = [500, 1000, 2500, 5000]
    N = 15   #trim this range edge voxel

    # for bth in bthlist:
    #     bn_path = os.path.join(inpath + f"\\bn({bth})")
    rootlist = glob.glob(f'{inpath}/*.vti')
    for root in rootlist:
        print(root)
        bnarray, spa, ori = vtk_data_loader(root)
    #     print("loading vti...")
    #     In_Img = os.path.join(inpath + filename + ".vti")
    #     print(In_Img)
    #     array, spa, ori = vtk_data_loader(In_Img)

    #     #mask
    #     #output = mask_processing(mask_Img, array)

    #     #trim
    #     #output = triming(array, N)
    #     #binarize
    #     bnarray = binarize(array, bth)
    #     print(bnarray.dtype, bnarray.shape)


        for vth in vthlist:
            #volume_processing
            print("volume processing")
            output, __ = volume_processing(bnarray, th=vth)
            print(output.dtype, output.shape)
            #output, __ = volume_processing_rev(array, th=vth)

            print("saving as vti...")
            output = numpy_to_vtk(output, spa, ori)
            filename = root.replace(inpath, "")
            filename = filename.replace(".vti", "")
            outroot = os.path.join(outpath  + f"\\frg(1,17,2)_bn(20)_VP({vth})" + filename + f"_vp({vth}).vti")
            print(outroot)
            save_vtk(output, outroot)


import vtk
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
from vtkmodules.util import numpy_support
import numpy as np
from skimage.filters import frangi
from scipy.ndimage import label
import cv2
from scipy import ndimage
import glob

inpath = 'd:\\takahashi_k\\preprocessing\\US(3D)\\Kaho_0085\\resize\\'
outpath = inpath
filename = "Origin"
inroot = os.path.join(inpath + filename + ".vti")
outroot = os.path.join(outpath + filename + "_frg(1-10)connected.vti")

ComputeSigma = False
sigmas = range(1, 11, 1)
compound = True
trim = 15
vp1 = 1000
vp2 = 500
bn = 10

def normalizetion(array):
    print("normalization")
    array = array * (255/array.max())
    array = array.astype(np.int8)
    return(array)

def labeling(array):
    #labeling
    str_3D = np.array([[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                        [[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]],
                        [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]], dtype='uint8')
    labeledarray, N = label(array, structure = str_3D)

    return(labeledarray, N)

def compound(allarray, maxarray):
    #labeling
    labeled_allarray, Nall = labeling(allarray)
    labeled_maxarray, Nall = labeling(maxarray)

    compoundarray = maxarray

    for N in range(1, Nall+1):
        extracted_image = (labeled_allarray==N)
        if np.sum(extracted_image*labeled_maxarray) != 0:
            compoundarray += extracted_image*allarray
    
    return(compoundarray)


def triming(array):
    #triming the edge
    print("triming the edge")
    dim = array.shape
    x = dim[0]
    y = dim[1]
    z = dim[2]
    
    array[:trim, :, :] = 0
    array[:, :trim, :] = 0
    array[:, :, :trim] = 0
    
    array[x-trim:, :, :] = 0
    array[:, y-trim:, :] = 0
    array[:, :, z-trim:] = 0
    return array

def OTSU(labeledarray, N):
    areas = []
    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        areas.append(count)

    areas = np.array(areas)
    areas = np.uint16(areas)
    th, __ =  cv2.threshold(areas, 0, 1, cv2.THRESH_OTSU)     #大津の2値化
    return(th)


def volume_processing(array, th = None):
    print("volume processing")
    labeledarray, N = labeling(array)

    if th == None:                      #大津の2値化による閾値の設定
        th = OTSU(labeledarray, N)
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
        th = OTSU(labeledarray, N)
        print(f"th by OTSU = {th}")

    output = np.zeros_like(array)

    areas = ndimage.sum(array, labeledarray, range(N + 1))
    mask_area = areas < th                                 
    remove_voxel = mask_area[labeledarray]
    labeledarray[remove_voxel] = 0
    output = (labeledarray > 0).astype(np.int8)
    output = output * array
    return(output, th)

def binarizaiton(array):
    print("binarization")
    compare = array > bn
    output = compare.astype(np.int8)
    return(output)

def compound_array(allarray, maxarray):
    #labeling
    labeled_allarray, Nall = labeling(allarray)
    labeled_maxarray, Nall = labeling(maxarray)

    compoundarray = maxarray

    for N in range(1, Nall+1):
        extracted_image = (labeled_allarray==N)
        if np.sum(extracted_image*labeled_maxarray) != 0:
            compoundarray += extracted_image*allarray
    
    return(compoundarray)

def ComputeSigmaBySpacing(spa):
    sgm = int(10/(2*max(spa)))
    return(sgm)

if __name__ == '__main__':
    #input vti-data
    print("loading vti file...")
    print(inroot)
    array, spa, ori = vtk_data_loader(inroot)

    if ComputeSigma:
        sigmas = ComputeSigmaBySpacing(spa)

    #frangi filter
    print("frangi filter")
    output = np.zeros_like(array)
    maxS = max(sigmas)
    for sigma in sigmas:
        temp = frangi(array, sigmas=range(sigma, sigma+1, 1), black_ridges=True)
        if compound and sigma == maxS:
            maxarray = frangi(array, sigmas=range(maxS, maxS+1, 1), black_ridges=True)
        output = np.maximum(output, temp)
    
    if compound:
        output = compound_array(output, maxarray)

    #normalization
    output = normalizetion(output)

    #triming the edge
    # output = triming(output)

    #volume processing
    #第二変数で閾値を設定しない場合、大津の2値化で求められた閾値により体積処理が行われる
    #output, th = volume_processing(output)
    # output, th = volume_processing_rev(output)

    #binalize
    # output = binarizaiton(output)

    #volume processing
    # output = volume_processing(output, vp2)
    # output = volume_processing_rev(output, vp2)

    #save as vti
    print("saving as vti...")
    output = numpy_to_vtk(output, spa, ori)
    print(outroot)
    save_vtk(output, outroot)


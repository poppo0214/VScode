import itk
import vtk
import cv2
from skimage.filters import frangi
from skimage.morphology import skeletonize_3d
import os
import numpy as np
from vtkmodules.util import numpy_support
from utils.labeling import labeling
from scipy import ndimage


def frangi_max_all_sigma(array, sgm, Maxs, dark):
    #apply frangi filter
    Allsigma = frangi(array, sigmas=sgm, black_ridges=dark)
    Maxsigma = frangi(array,sigmas=[Maxs],black_ridges=dark)

    # for N in sgm:
        # Allsigma = np.zeros_like(array)
    #     output = frangi(array,sigmas=[N],alpha=0.5,beta=0.5,gamma=15,black_ridges=dark)
    #     output = norm_0_255(output)
    #     compare = output > Allsigma                             #output > Allsigma:1, output =< Allsigma:0
    #     inverse_comp = (~compare)                                 #compareを反転(0->1, 1->0)
    #     Allsigma = compare*output + Allsigma*inverse_comp       #Allsigmaよりoutputが大きい輝度値はoutputに置き換え、それ以外はAllsigmaのまま
    #     if N==Maxs:
    #         Maxsigma = output
    
    return(Allsigma, Maxsigma)

def norm_0_255(array):
    output = array*(255/array.max()) 
    output = output.astype(np.int8)
    return (output)

def triming(array, N):
    dim = array.shape
    x = dim[0]
    y = dim[1]
    z = dim[2]
    
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

def volume_processing(array):
    labeledarray, N = labeling(array)

    areas = []
    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        areas.append(count)

    areas = np.array(areas)
    areas = np.uint16(areas)                            

    th, __ =  cv2.threshold(areas, 0, 1, cv2.THRESH_OTSU)     #大津の2値化

    output = np.zeros_like(array)

    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        if count >= th:
            temp = np.where(labeledarray == num , 1, 0)
            output = output + temp

    output = output * array
    return(output)

def volume_processing_rev(array):
    labeledarray, N = labeling(array)

    areas = ndimage.sum(array, labeledarray, range(N+1))

    areas = np.array(areas)
    areas = np.uint16(areas)                            

    th, __ =  cv2.threshold(areas, 0, 1, cv2.THRESH_OTSU)     #大津の2値化

    output = np.zeros_like(array)

    mask_area = areas < th                                 
    remove_voxel = mask_area[labeledarray]
    labeledarray[remove_voxel] = 0
    output = (labeledarray > 0).astype(np.int8)
    output = output * array
    return(output)


def volume_processing_byTH(array, th):
    output = np.zeros_like(array)
    labeledarray, N = labeling(array)

    #count largest sigma vessel label
    sizes = ndimage.sum(array, labeledarray, range(N+1))
    
    # Noise reduction at a certain threshold(n_th) in largest sigma vessel
    mask_size = sizes < th                                 
    remove_voxel = mask_size[labeledarray]
    labeledarray[remove_voxel] = 0
    output = (labeledarray > 0).astype(np.int8)

    # for num in range(1,N+1):
    #     count = np.count_nonzero(labeledarray == num)
    #     if count > th:
    #         temp = np.where(array == num, 1, 0)
    #         output = output + temp

    return(output)

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

def compound_with_doppler(allarray, maxarray, darray):
    #labeling
    labeled_allarray, Nall = labeling(allarray)
    labeled_maxarray, Nmax = labeling(maxarray)

    max_d = np.zeros_like(maxarray)

    for i in range(1, Nmax+1):
        extracted_max = (labeled_maxarray==i)
        if np.sum(extracted_max*darray) != 0:
            max_d += extracted_max*maxarray
    
    # # array to image
    # output = numpy_to_vtk(max_d)
    # # save output file(.vti)
    # Out_Img = "D:\\takahashi_k\\frangi\\withdoppler\\MaxandDoppler.vti"
    # save_vtk(output, Out_Img)

    compoundarray = max_d
    for j in range(1, Nall+1):
        extracted_all = (labeled_allarray==j)
        if np.sum(extracted_all*max_d) != 0:
            compoundarray += extracted_all*allarray
    
    return(compoundarray)    

def binarizaiton(array, th):
    compare = array > th
    output = compare.astype(np.int8)
    return(output)


def thin_junstion(array):
    labeledarray, N = labeling(array)

    for num in range(1, N+1):
        extracted_image = (labeledarray==num)    
        centerline = skeletonize_3d(extracted_image)
        (x, y, z) = extracted_image.shape

        vessel = False
        for i in range(1, z, 1):
            for j in range(1, y, 1):
                for k in range(1, x, 1):
                    count = 0
                    for a in range(-1, 2, 1):
                        for b in range(-1, 2, 1):
                            for c in range(-1, 2, 1):
                                count += centerline[k+c, j+b, i+a]
                    if count >= 4:
                        vessel = True
        if vessel:
            temp = np.where(labeledarray == num , 1, 0)
            output = output + temp

    output = output * array

    return(output)
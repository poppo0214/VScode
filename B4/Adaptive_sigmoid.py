from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import numpy as np
from skimage.exposure import adjust_gamma, adjust_log, adjust_sigmoid
import kmeans1d
import math
import glob



def adaptive_sigmoid(array):
    """An Improved Fuzzy Connectedness Method for Automatic Three-Dimensional Liver Vessel Segmentation in CT Images"""
    print("adjusting by sigmoid(ShuicaiWu)")
    flatten_array = array.flatten()
    #k = 5                       #クラスター数（背景、低輝度血管（静脈系）、高輝度血管（門脈系）、肝実質、横隔膜）
    k = 6                       #クラスター数（背景、胆のう、低輝度血管（静脈系）、高輝度血管（門脈系）、肝実質、横隔膜）
    #clusters->arrayと同じサイズで、各voxelのラベル　#centroids->各クラスターの中心値
    clusters, centroids = kmeans1d.cluster(flatten_array, k)            
    print(centroids)
    if k==5:
        alpha = (centroids[2]-centroids[1])/2
        beta = (centroids[2]+centroids[1])/2
    elif k==6:
        alpha = (centroids[3]-centroids[2])/2
        beta = (centroids[3]+centroids[2])/2
    output = (np.amax(array) - np.amin(array))/(1 + math.e**((beta-array)/alpha))
    return(output)


if __name__ == '__main__':
    In_path = r"d:\takahashi_k\database\us\senior\origin"
    Out_path = r"d:\takahashi_k\preprocessing\us\senior\AS"
    
    rootlist = glob.glob(f'{In_path}/*Origin.vti')

    for root in rootlist:
        print(root)
        array, spa, ori = vtk_data_loader(root)
        output = adaptive_sigmoid(array)
        
        filename = root.replace(In_path, "")
        filename = filename.replace(".vti", "")
        Out_root = os.path.join(Out_path + filename + "_AS(6).vti")
        print(Out_root)
        output = numpy_to_vtk(output, spa, ori)
        save_vtk(output, Out_root)
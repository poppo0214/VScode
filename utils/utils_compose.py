from scipy.ndimage import label
from scipy import ndimage
import numpy as np
import itk
from utils_vtk import numpy_to_vtk, save_vtk, vtk_data_loader

def Read(Pass):
    input = itk.imread(Pass)
    array = itk.GetArrayFromImage(input)
    Np_Array = np.array(array)
    #nichika
    thresh = 0
    im_bool = (Np_Array > thresh) * 1
    return np.array(im_bool)

def label_procedure(MaxSigmaImg, AllSigmaImg, SavePath):
    str_3D = np.array([[[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]],
                       [[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]],
                       [[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]], dtype='uint8')
    MaxLabeled, MaxNumFeatures = label(MaxSigmaImg, structure = str_3D)
    AllLabeled, AllNumFeatures = label(AllSigmaImg, structure = str_3D)
    #number of label
    # print("number of labels in image_max: ", MaxNumFeatures)
    # print("number of labels in image_all: ", AllNumFeatures)
        
    #count OR vessel label
    AllCntVoxel = ndimage.sum(AllSigmaImg, AllLabeled, range(AllNumFeatures + 1))
    
    #count largest sigma vessel label
    MaxCntVoxel = ndimage.sum(MaxSigmaImg, MaxLabeled, range(MaxNumFeatures + 1))
    
    # Noise reduction at a certain threshold(n_th) in largest sigma vessel
    MaskNoiseReduction = MaxCntVoxel < 1000                                 #体積処理の閾値
    RemoveVoxel = MaskNoiseReduction[MaxLabeled]
    MaxLabeled[RemoveVoxel] = 0

    # fusion largest sigma vessel after noise reduction and OR vessel
    RenketsuResult = AllLabeled * (MaxLabeled > 0)

    for i in range(0, AllNumFeatures+1):
        if np.sum(RenketsuResult[RenketsuResult == i]) == 0:
            AllLabeled[AllLabeled == i] = 0

    ComposeResult = (AllLabeled > 0) * 255
    
    #mask procedure
    y_img = "./US_Mask_Image.vti"
    Y_img = vtk_data_loader(y_img)
    MaskComposeResult = ComposeResult * Y_img
    
    VtkComposeResult = numpy_to_vtk(MaskComposeResult)
    save_vtk(VtkComposeResult, SavePath)
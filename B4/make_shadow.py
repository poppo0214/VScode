from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import numpy as np
import cv2
from skimage.filters import gaussian


In_path = "d:\\takahashi_k\\database\\simulation\\vessel\\v2\\"
Out_path = In_path
In_name = "vessel_v2_rev_pad_us_noise"


if __name__ == '__main__':
    In_root = os.path.join(In_path + In_name + '.vti')
    Out_root = os.path.join(Out_path + In_name + '_shadow.vti')
    print(In_root)

    print("loading vti file...")
    array, spa, ori = vtk_data_loader(In_root)

    #array = cv2.imread(In_root, cv2.IMREAD_GRAYSCALE)

    shape = array.shape
    print(shape)
    (Ex, Ey, Ez) = shape

    #xy平面のx方向の両端30％に最大-50のノイズをのせる
    width = int(Ex * 0.3)       #影がつく部分の幅（voxel数）
    step = 50/width             #影のグラデーションの輝度値が1voxel毎でどれくらい変わるか

    #shadow = []
    shadow_array = np.zeros_like(array)
    temp = 50
    for i in range(0, width, 1):
        #shadow.append(temp)
        shadow_array[i, :, :] = temp
        temp -= step
    #shadow += [0] * (Ex-2*width)        #影がつかない部分を追加
    temp = step
    for j in range(Ex-width, Ex, 1):
        #shadow.append(temp)
        shadow_array[j, :, :] = temp
        temp += step
    
    output = array - shadow_array

    output = np.where(output < 0, 0, output)            #0より小さいボクセルは0に置換
    output = np.where(output > 255, 255, output)        #255より大きいボクセルは255に置換
    # # output = 255*(output-np.amin(output))/(np.amax(output)-np.amin(output))               #0-255で正規化

    #8bitに変換
    output_8bit = output.astype(np.uint8)
    print(output_8bit.dtype)
    print(np.amin(output_8bit), np.amax(output_8bit))
    print(output_8bit.shape)

    #save as vti
    print("saving as vti...")
    output = numpy_to_vtk(output_8bit, spa, ori)
    save_vtk(output, Out_root)
    #cv2.imwrite(Out_root, output_8bit)
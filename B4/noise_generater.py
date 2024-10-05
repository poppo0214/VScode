from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import numpy as np
import cv2
from skimage.filters import gaussian

filename = "gauss(non)_frg(1,2,1)"
alpha = 1               #ガンマ分布の形状母数（shape/s/α）大きいほど明るいノイズが増える
beta = 1/alpha          #ガンマ分布の尺度母数（scale/r/β） beta = 1\alpha
mu = 0                  #ガウス分布の平均（loc/μ）
sgm2 = 10               #ガウス分布の標準偏差シグマ（scale/σ）      大きいほどノイズが大きくなる
sgm1 = 1                #平滑化フィルタとしてのガウシアンフィルタのシグマ
back = 150              #背景の輝度値
vessel = 70             #血管の輝度値
tannnou = 45            #胆のうの輝度値
oukakumaku = 200        #横隔膜の輝度値

In_path = "d:\\takahashi_k\\database\\simulation\\vessel\\v2\\"
Out_path = In_path
In_name = "vessel_v2_rev_pad"
Out_name =  f'vessel_v2_rev_pad_us_noise'

def speckle(shape):
    #output = np.random.gamma(alpha, beta, shape)
    output = np.random.normal(mu, 1, shape)
    return(output)


def gauss(shape):
    output = np.random.normal(mu, sgm2, shape)
    return(output)


if __name__ == '__main__':
    In_root = os.path.join(In_path + In_name + '.vti')
    Out_root = os.path.join(Out_path + Out_name + '.vti')
    print(In_root)

    print("loading vti file...")
    array, spa, ori = vtk_data_loader(In_root)
    shape = array.shape

    array = np.where(array == 1, vessel, back)     #背景と血管部分が指定の値になるように乗算
    # array = np.where(array == 0, 150, array)           #背景の輝度値を150
    # array = np.where(array == 1, 200, array)         #横隔膜の輝度値を200
    # array = np.where(array == 2, 70, array)          #血管の輝度値を70
    # output = np.where(array == 3, 45, array)          #胆のうの輝度値を30

    print("add speckle noise")
    speckle_array = speckle(shape)
    #output = array + array * speckle_array                                       #乗算ノイズ
    output = array + (array * speckle_array/4) - np.mean(array * speckle_array)            #乗算ノイズ

    print("add gauss noise")
    gauss_array = gauss(shape)
    output = output + gauss_array            #加算ノイズ

    output = gaussian(output, sigma=sgm1)   #ノイズを滑らかにする

    output = np.where(output < 0, 0, output)            #0より小さいボクセルは0に置換
    output = np.where(output > 255, 255, output)        #255より大きいボクセルは255に置換
    # # output = 255*(output-np.amin(output))/(np.amax(output)-np.amin(output))               #0-255で正規化

    # #8bitに変換
    # output_8bit = output.astype(np.uint8)
    # print(output_8bit.dtype)
    # print(np.amin(output_8bit), np.amax(output_8bit))

    #save as vti
    print("saving as vti...")
    output = numpy_to_vtk(output, spa, ori)
    save_vtk(output, Out_root)
    #cv2.imwrite(Out_root, output_8bit)
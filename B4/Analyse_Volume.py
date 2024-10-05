from scipy.ndimage import label
from utils.utils_vtk import vtk_data_loader
import os
import numpy as np
import itk
import matplotlib.pyplot as plt

inpath = "D:\\takahashi_k\\simulation\\add_noise\\vessel\\"
outpath = inpath
filename = "noise_close(5)_mask(5)"
mhd = False
log = False                                                #出力するヒストグラムの縦軸（連結成分数）の対数をとるかどうか   

#get histgram
def histogram(array, name, histlog, N):
    # Graph Settings           
    plt.title(f"distribution of {name}'s volume\nthe number of connected components is {N}", fontsize=15)  
    plt.xlabel("volume of connected components", fontsize=12)   
    plt.ylabel("the number of connected components", fontsize=12) 

    n, bins, _ = plt.hist(array, bins=np.logspace(0, 7, 40), log = histlog)               #n : 配列または配列のリスト,ヒストグラムのビンの値    bins：いくつの階級に分割するか（logスケール　10^0 ~ 10^2を50この階級に分ける）
    plt.xscale("log")

    xs = (bins[:-1] + bins[1:])/2
    ys = n.astype(int)

    for x, y in zip(xs, ys):
        plt.text(x, y, str(y), horizontalalignment="center", size = 6)
    
    
    if histlog:
        plt.savefig(os.path.join(outpath + f"\\histogram_{name}'s volume(ylog).png"))
    else:
        plt.savefig(os.path.join(outpath + f"\\histogram_{name}'s volume.png"))


if __name__ == '__main__':

    print("loading vti...")
    In_Img = os.path.join(inpath  + filename + ".vti")
    array, _, _ = vtk_data_loader(In_Img)

    # #nichika
    # thresh = 0
    # #compare = np.full_like(array, thresh)
    # compare = array > thresh
    # e = compare.astype(np.int8)

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

    labeledarray = labeledarray[labeledarray != 0]                      #計算量を少なくするために０以外の値を抽出

    #calculate volume
    print("calculatnig volume...")
    areas = []
    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        areas.append(count)
    print("finished calculating")
    print(max(areas), N)

    #histogram of connected components's volume
    histogram(areas, filename, log, N)
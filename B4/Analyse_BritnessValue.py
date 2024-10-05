import numpy as np
import itk
import matplotlib.pyplot as plt
import os
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk

comperison = True       #2つのデータに対して分析を行うかどうか
log = False             #出力するヒストグラムの縦軸（ボクセル数）の対数をとるかどうか
mhd = False

inpath1 = "D:/takahashi_k/frangi/test/"
filename1 = "patient1-Origin_ROI"

if comperison:
    inpath2 = "D:/takahashi_k/frangi/test/"
    filename2 = "patient3-Origin_ROI"

#get date of array 
def arraydeta(array):
    shape = array.shape
    size = array.size
    type = array.dtype
    max = array.max()
    min = array.min()
    deta = f"shape of 3Dimage is {shape}.\n\
        the number of voxel is {size}.\n\
        deta type is {type}.\n\
        maximum brightness value is {max}.\n\
        minimum brightness value is {min}\n"
    return deta

#get histgram
def histogram(array, name, histlog):
    # Graph Settings      
    plt.title(f"distribution of {name}", fontsize=15)  
    plt.xlabel("brightness value", fontsize=12)   
    plt.ylabel("the number of voxel", fontsize=12) 

    rearray = array.flatten()                             #3次元配列を1次元に平滑化
    n, bins, _ = plt.hist(rearray, bins=np.linspace(0, 255, 20),log = histlog)               #n : 配列または配列のリスト,ヒストグラムのビンの値    bins：いくつの階級に分割するか
    
    xs = (bins[:-1] + bins[1:])/2
    ys = n.astype(int)

    for x, y in zip(xs, ys):
        plt.text(x, y, str(y), horizontalalignment="center", size = 6)
    
    if histlog:
        plt.savefig(f"D:/takahashi_k/frangi/brightness/histogram_{name}(log).png")
    else:
        plt.savefig(f"D:/takahashi_k/frangi/brightness/histogram_{name}.png")

def histogram(array, name, histlog):
    # Graph Settings            
    plt.title(f"distribution of {name}", fontsize=20)  
    plt.xlabel("brightness value", fontsize=12)   
    plt.ylabel("the number of voxel", fontsize=12) 

    rearray = array.flatten()                             #3次元配列を1次元に平滑化
    n, bins, _ = plt.hist(rearray, bins=np.linspace(0, 255, 20), 
    log = histlog)               #n : 配列または配列のリスト,ヒストグラムのビンの値    bins：いくつの階級に分割するか
    
    xs = (bins[:-1] + bins[1:])/2
    ys = n.astype(int)

    for x, y in zip(xs, ys):
        plt.text(x, y, str(y), horizontalalignment="center", size=6, rotation=90)
    
    if histlog:
        plt.savefig(f"D:/takahashi_k/frangi/brightness/histogram_{name}(log).png")
    else:
        plt.savefig(f"D:/takahashi_k/frangi/brightness/histogram_{name}.png")

def com_histogram(array1, array2, name1, name2, histlog):
    # Graph Settings
    #plt.xlim(0, 255)                 
    plt.title(f"distribution of\n{name1} and {name2}", fontsize=15)  
    plt.xlabel("brightness value", fontsize=12)   
    plt.ylabel("the number of voxel", fontsize=12) 

    rearray1 = array1.flatten()                             #3次元配列を1次元に平滑化
    rearray2 = array2.flatten() 

    bins = np.linspace(0, 255, 20)
    plt.hist([rearray1, rearray2], bins, log = histlog, label= [f"{name1}", f"{name2}"])               #n : 配列または配列のリスト,ヒストグラムのビンの値    bins：いくつの階級に分割するか
    plt.legend(loc='upper left')

    if histlog:
        plt.savefig(f"D:/takahashi_k/frangi/brightness/histogram_{name1}_and_{name2}(log).png")
    else:
        plt.savefig(f"D:/takahashi_k/frangi/brightness/histogram_{name1}_and_{name2}.png")


if __name__ == '__main__':

    if mhd:
        print("loading mhd...")
        In_Img1 = os.path.join(inpath1 + filename1 + ".mhd")

        #image to array
        input1 = itk.imread(In_Img1)
        array1 = itk.GetArrayFromImage(input1)

        if comperison:
            In_Img2 = os.path.join(inpath2 + filename2 + ".mhd")
            input2 = itk.imread(In_Img2)
            array2 = itk.GetArrayFromImage(input2)


    else:
        print("loading vti...")
        In_Img1 = os.path.join(inpath1 + filename1 + ".vti")
        array1 = vtk_data_loader(In_Img1)

        if comperison:
            In_Img2 = os.path.join(inpath2 + filename2 + ".vti")
            array2 = vtk_data_loader(In_Img2)

    #crate txt
    if comperison == False:
        f = open(f"D:/takahashi_k/frangi/brightness/analyse_{filename1}.txt", 'a')
        arraydeta1 = arraydeta(array1)
        f.write("array1:\n")
        f.write(arraydeta1)
        f.close

    else:
        f = open(f"D:/takahashi_k/frangi/brightness/analyse_{filename1}_{filename2}.txt", 'a')
        arraydeta1 = arraydeta(array1)
        f.write("array1:\n")
        f.write(arraydeta1)
        arraydeta2 = arraydeta(array2)
        f.write("array2:\n")
        f.write(arraydeta2)
        f.close() 

    #histogram
    if comperison == False:
        histogram(array1, filename1, log)

    else:
        com_histogram(array1, array2, filename1, filename2, log)
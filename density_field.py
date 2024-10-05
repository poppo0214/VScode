import os
import sys
import copy
import numpy as np
import open3d as o3d
import glob
import vtk
from vtkmodules.util import numpy_support
from skimage.morphology import opening, closing, cube
from utils.utils_vtk import numpy_to_vtk, save_vtk
import joblib as jl

def calculate_density(point_array, spacing):
    """
    点群密度の計算
    """
    x_max = np.max(point_array[:, 0:1])
    x_min = np.min(point_array[:, 0:1])
    y_max = np.max(point_array[:, 1:2])
    y_min = np.min(point_array[:, 1:2])
    z_max = np.max(point_array[:, 2:])        
    z_min = np.min(point_array[:, 2:])
    
    #boundsとspacingからextentsを計算
    origin = [x_min, y_min, z_min]
    x_extent = round((x_max-x_min)/spacing[0])+1
    y_extent = round((y_max-y_min)/spacing[1])+1
    z_extent = round((z_max-z_min)/spacing[2])+1
    # print(f"x\tfrom {x_min} to {x_max}: {x_extent}")
    # print(f"y\tfrom {y_min} to {y_max}: {y_extent}")
    # print(f"z\tfrom {z_min} to {z_max}: {z_extent}")

    num = len(point_array)            #vtpのポイント数を取得

    #print(f"converted vti origin:{origin}\nconverted vti spacing:{spacing}")
    #boundsとspacingからdimentionsを計算して、空の配列を生成
    # x_dim = round(x_extent/spacing[0]) + 1
    # y_dim = round(y_extent/spacing[1]) + 1
    # z_dim = round(z_extent/spacing[2]) + 1
    # dimentions = (x_dim, y_dim, z_dim)
    #print("dimentions:", dimentions)
    density_array = np.zeros((x_extent, y_extent, z_extent))

    #polyデータの座標を取得して、近くの座標に対応する配列を1にする
    for i in range(num):
        point = point_array[i]
        index = []
        index.append(round((point[0]-x_min)/spacing[0]))
        index.append(round((point[1]-y_min)/spacing[1]))
        index.append(round((point[2]-z_min)/spacing[2]))
        density_array[index[0], index[1], index[2]] += 1
    
    #正規化
    #print(np.max(density_array))
    #density_array = density_array*(255/np.max(density_array))
    print(f"density_array:\n{density_array}")

    return(density_array, spacing, origin)

def calculate_densityFiled(density_array):
    """
    点群密度場の計算
    """
    #print(density_array.shape)
    i = density_array.shape[0]
    j = density_array.shape[1]
    k = density_array.shape[2]
    dnsFiled_array = np.zeros_like(density_array)
    
    #3次元カーネルの生成
    ksz = 10
    kernel_3d = np.zeros((ksz*2+1, ksz*2+1, ksz*2+1))
    for c in range(-ksz, ksz+1):
        for b in range(-ksz, ksz+1):
            for a in range(-ksz, ksz+1):
                if a==0 and b==0 and c==0:
                    kernel_3d[a+ksz,b+ksz,c+ksz] = 0
                else:
                    kernel_3d[a+ksz,b+ksz,c+ksz] = 1/(a*a + b*b + c*c)
    #print(f"kernel\n{kernel_3d}")

    #padding
    # output = np.pad(array, [(前0画像数, 後0画像数), (上0埋め数, 下0埋め数), (左0埋め数, 右0埋め数)], 'constant')
    pad_density_array = np.pad(density_array, [(ksz, ksz), (ksz, ksz), (ksz, ksz)], 'constant')
    print(f"padding: {density_array.shape}->{pad_density_array.shape}")
    #print(density_array)
    for z in range(k):
        for y in range(j):
            for x in range(i):
                #注目ボクセル：[i,j,k]
                cutoff_array = pad_density_array[x:x+ksz*2+1, y:y+ksz*2+1, z:z+ksz*2+1]
                # kernel_3d = copy.deepcopy(kernel_3d)
                # kernel_3d = np.pad(kernel_3d, [(x, i-x-1), (y, j-y-1), (z, k-z-1)], 'constant')
                # if x==0 and y==0 and z==0:
                #     print(cutoff_array.shape)
                #     print(cutoff_array)
                    # print(kernel_3d.shape)
                    # print(kernel_3d)
                # #print(x,y,z)
                # #dnsFiled_array[i,j,k] = np.sum(density_array * kernel_3d)
                # #dnsFiled_array[i,j,k] = np.sum(np.ravel(density_array) * np.ravel(kernel_3d))
                # #d = np.sum(density_array * kernel_3d)
                result = np.sum(cutoff_array * kernel_3d)
                #print(f"({x}, {y}, {z}) -> {result}")
                dnsFiled_array[x,y,z] = result

                # dnsFiled_array[x,y,z] = np.sum(cutoff_array * kernel_3d)


    #正規化
    # print(np.max(dnsFiled_array))
    # density_array = dnsFiled_array*(255/np.max(dnsFiled_array))
    return(dnsFiled_array)

def normalizetion(array, maxV):
    print(f"normalizetion: {np.min(array)} ~ {np.max(array)} -> {np.min(array)} ~ {np.max(array) * (255/np.max(array))}")
    afterarray = array * (255/np.max(array))
    return afterarray  





if __name__ == "__main__":
    in_path = r"D:\takahashi_k\test\density"
    in_name = r"vessel(filled)"
    out_path = r"D:\takahashi_k\test\density"
    spacing = [0.5, 0.5, 0.5]
    maxV = 255

    in_root = os.path.join(in_path + "\\" + in_name + ".pcd")
    density_root = os.path.join(out_path + "\\" + in_name + "_dencity.vti")
    dnsFiled_root = os.path.join(out_path + "\\" + in_name + "_dencity_field.vti")

    in_pcd =  o3d.io.read_point_cloud(in_root)
    point_array = np.asarray(in_pcd.points)
    density_array, spacing, origin = calculate_density(point_array, spacing)
    density_array_norm = normalizetion(density_array, maxV)
    density_data = numpy_to_vtk(density_array_norm, spacing, origin)
    save_vtk(density_data, density_root)

    dnsFiled_array = calculate_densityFiled(density_array)
    dnsFiled_array_norm = normalizetion(dnsFiled_array, maxV)
    dnsFiled_data = numpy_to_vtk(density_array_norm, spacing, origin)
    save_vtk(dnsFiled_data, dnsFiled_root)
    




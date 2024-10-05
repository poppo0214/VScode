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
import itertools
import joblib as jl
import multiprocessing as mp
from multiprocessing import shared_memory

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
    x_extent = round((x_max-x_min)/spacing[0])
    y_extent = round((y_max-y_min)/spacing[1])
    z_extent = round((z_max-z_min)/spacing[2])
    print(f"x\tfrom {x_min} to {x_max}: {x_extent}")
    print(f"y\tfrom {y_min} to {y_max}: {y_extent}")
    print(f"z\tfrom {z_min} to {z_max}: {z_extent}")

    num = len(point_array)            #vtpのポイント数を取得

    #print(f"converted vti origin:{origin}\nconverted vti spacing:{spacing}")
    #boundsとspacingからdimentionsを計算して、空の配列を生成
    x_dim = round(x_extent/spacing[0]) + 1
    y_dim = round(y_extent/spacing[1]) + 1
    z_dim = round(z_extent/spacing[2]) + 1
    dimentions = (x_dim, y_dim, z_dim)
    print("dimentions:", dimentions)
    density_array = np.zeros(dimentions)

    #polyデータの座標を取得して、近くの座標に対応する配列を1にする
    for i in range(num):
        point = point_array[i]
        index = []
        index.append(round((point[0]-x_min)/spacing[0]))
        index.append(round((point[1]-y_min)/spacing[1]))
        index.append(round((point[2]-z_min)/spacing[2]))
        density_array[index[0], index[1], index[2]] += 1
    
    #正規化
    print(np.max(density_array))
    density_array = density_array*(255/np.max(density_array))

    return(density_array, spacing, origin)

def process(value, shared_mem_name):
    existing_shm = shared_memory.SharedMemory(name=shared_mem_name)
    img_np = np.ndarray(shape, dtype=np.uint8, buffer=existing_shm.buf)
    kernel_3d = copy.deepcopy(origin_kernel_3d_mp)
    kernel_3d = np.pad(kernel_3d, [(value[0], i_mp-value[0]-1), (value[1], j_mp-value[1]-1), (value[2], k_mp-value[2]-1)], 'constant')
    return np.sum(density_array_mp * kernel_3d)


def calculate_densityFiled(density_array):
    """
    点群密度場の計算
    """
    print(density_array.shape)
    i = density_array.shape[0]
    j = density_array.shape[1]
    k = density_array.shape[2]
    dnsFiled_array = np.zeros_like(density_array)
    #3次元カーネルの生成
    kernel_size = 2
    origin_kernel_3d = np.zeros((kernel_size*2+1, kernel_size*2+1, kernel_size*2+1))
    for c in range(-kernel_size, kernel_size+1):
        for b in range(-kernel_size, kernel_size+1):
            for a in range(-kernel_size, kernel_size+1):
                if a==0 and b==0 and c==0:
                    origin_kernel_3d[a+kernel_size,b+kernel_size,c+kernel_size] = 0
                else:
                    origin_kernel_3d[a+kernel_size,b+kernel_size,c+kernel_size] = 1/(a*a + b*b + c*c)
    print(origin_kernel_3d)

    #padding
    # output = np.pad(array, [(前0画像数, 後0画像数), (上0埋め数, 下0埋め数), (左0埋め数, 右0埋め数)], 'constant')
    density_array = np.pad(density_array, [(kernel_size, kernel_size), (kernel_size, kernel_size), (kernel_size, kernel_size)], 'constant')
    
    # for z in range(k):
    #     for y in range(j):
    #         for x in range(i):
                #注目ボクセル：[i,j,k]
                # kernel_3d = copy.deepcopy(origin_kernel_3d)
                # kernel_3d = np.pad(kernel_3d, [(x, i-x-1), (y, j-y-1), (z, k-z-1)], 'constant')
                # if x==0 and y==0 and z==0:
                #     print(density_array.shape)
                #     print(kernel_3d.shape)
                # dnsFiled_array[x,y,z] = np.sum(density_array * kernel_3d)
    
    
    density_array_shm = shared_memory.SharedMemory(chreat=True, size=np.prod(density_array.shape) * np.dtype(np.float64).itemsize)
    density_array_sh = np.ndarray(density_array.shape, dtype=np.float64, buffer=density_array_shm.buf)
    origin_kernel_3d_shm = shared_memory.SharedMemory(chreat=True, size=np.prod(origin_kernel_3d.shape) * np.dtype(np.float64).itemsize)
    origin_kernel_3d_sh = np.ndarray(origin_kernel_3d.shape, dtype=np.float64, buffer=origin_kernel_3d_shm.buf)

    i_mp = mp.Value("i", i)
    j_mp = mp.Value("i", j)
    k_mp = mp.Value("i", k)

    p = mp.Pool(processes=mp.cpu_count())
    values = [(x, y, z) for x in range(i) for y in range(j) for z in range(k)]
    result = p.starmap(process, values)
    p.close()
    dnsFiled_array = np.array(result).reshape((i,j,k))


    #正規化
    print(np.max(dnsFiled_array))
    density_array = dnsFiled_array*(255/np.max(dnsFiled_array))
    return(dnsFiled_array)






if __name__ == "__main__":
    in_path = r"D:\takahashi_k\test\density"
    in_name = r"vessel(filled)"
    out_path = r"D:\takahashi_k\test\density"
    spacing = [0.5, 0.5, 0.5]

    in_root = os.path.join(in_path + "\\" + in_name + ".pcd")
    density_root = os.path.join(out_path + "\\" + in_name + "_dencity.vti")
    dnsField_root = os.path.join(out_path + "\\" + in_name + "_dencity_field.vti")

    in_pcd =  o3d.io.read_point_cloud(in_root)
    point_array = np.asarray(in_pcd.points)
    density_array, spacing, origin = calculate_density(point_array, spacing)
    density_data = numpy_to_vtk(density_array, spacing, origin)
    save_vtk(density_data, density_root)

    dnsField_array = calculate_densityFiled(density_array)
    dnsField_data = numpy_to_vtk(density_array, spacing, origin)
    save_vtk(dnsField_data, dnsField_root)
    




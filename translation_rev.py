import os
import numpy as np
import glob
import random
import math
import open3d as o3d
# def pcd_mtx(pcd_array, mtx):
#     print(pcd_array.shape, pcd_array)
#     paded_array = np.pad(pcd_array,((0,0),(0,1)),  mode="constant", constant_values=1)
#     print(paded_array.shape, paded_array)
#     paded_array = paded_array.T    
#     print(paded_array.shape, paded_array)

#     multied_array = np.dot(mtx, paded_array)
#     print(multied_array.shape, multied_array)
#     multied_array = multied_array.T
#     print(multied_array.shape, multied_array)
#     multied_array = multied_array[:, :3]
#     print(multied_array.shape, multied_array)
#     return multied_array


def pcd_mtx(pcd_array, mtx):
    # print("origin array")
    # print(pcd_array.shape)
    # print(pcd_array)
    paded_array = np.pad(pcd_array,((0,0),(0,1)),  mode="constant", constant_values=1)
    # print("paded array")
    # print(paded_array.shape)
    # print(paded_array)
    moved_list = []
    for i in range(paded_array.shape[0]):
        origin_point = paded_array[i, :].T
        moved_point = np.dot(mtx, origin_point)
        moved_list.append(moved_point)
        #print(origin_point, moved_point)
    moved_array = np.asarray(moved_list)[:, :3]
    # print("moved array")
    # print(moved_array.shape)
    # print(moved_array)
    # moved_array = moved_array.T
    # print("tenchied moved array") 
    # print(moved_array.shape)
    # print(moved_array)
    # moved_array = moved_array[:3, :]
    # print(moved_array.shape)
    # print(moved_array)
    return moved_array

"""複数のデータを複数の値で移動させる場合(pcd)"""
if __name__ == '__main__':
    Inpath = r"D:\takahashi_k\registration(model)\forUSE\original"
    Outpath = Inpath
    """ターゲット側は動かさなくてよい！！！"""
    Error_list = range(5, 21, 5)              #どれだけずらすか[mm]
    N = 10                                  #何データ作成するか

    rootlist = glob.glob(f'{Inpath}/*Y-deform_padding(20)_closing(10).pcd')
    for Inroot in rootlist:
        #pcdデータ読み込み,numpy配列に変換
        pcd_data = o3d.io.read_point_cloud(Inroot)
        #o3d.visualization.draw_geometries([pcd_data])
        pcd_array = np.asarray(pcd_data.points)
        filename = Inroot.replace(Inpath, "")
        filename = filename.replace(".pcd", "")
        filename = filename.replace("\\", "")
        print(filename)

        for Error in Error_list:
            Outpcd_path = os.path.join(Outpath + f"\\translation_{Error}mm")
            if not os.path.exists(Outpcd_path):
                os.mkdir(Outpcd_path)
            Outmtx_path = os.path.join(Outpcd_path + f"\\TrsMtx_GT")
            if not os.path.exists(Outmtx_path):
                os.mkdir(Outmtx_path)
            for num in range(0, N):        
                TrsMtx_root = os.path.join(Outmtx_path + f"\\{filename}_{Error}-{num}.tsv")           
                Outpcd_root = os.path.join(Outpcd_path + f"\\{filename}_{Error}-{num}.pcd")
                #各軸の並進移動量をランダムに決定する
                x_ran = random.random()
                y_ran = random.random()
                z_ran = random.random()
                D = math.sqrt(x_ran*x_ran + y_ran*y_ran + z_ran*z_ran)
                x_error = x_ran*Error/D
                y_error = y_ran*Error/D
                z_error = z_ran*Error/D
                D = math.sqrt(x_error*x_error + y_error*y_error + z_error*z_error)
                print(D)
                TrsMtx = np.asarray([[1,0,0,x_error],[0,1,0,y_error],[0,0,1,z_error],[0,0,0,1]])
                # print("translation mateix")
                # print(TrsMtx.shape)
                # print(TrsMtx)

                moved_array = pcd_mtx(pcd_array, TrsMtx)
                moved_pcd = o3d.geometry.PointCloud()
                moved_pcd.points = o3d.utility.Vector3dVector(moved_array)
                np.savetxt(TrsMtx_root, TrsMtx, delimiter='\t')
                o3d.io.write_point_cloud(Outpcd_root, moved_pcd) 
                



                
                

import numpy as np
import vtk
import os
from utils.utils_vtk import vtp_data_loader, save_vtp
import glob

def vtp_to_list_vti(polydata, root=None, spacing=None):
    #vtpからorigin（インデックスが0の点の座標）とboundsを取得
    bounds = polydata.GetBounds()                 #vtpの境界を取得
    origin = [bounds[0], bounds[2], bounds[4]]    #境界の端をoriginに指定
    num = polydata.GetNumberOfPoints()            #vtpのポイント数を取得
    point_list = []


    #print(f"converted vti origin:{origin}\nconverted vti spacing:{spacing}")
    #boundsとspacingからdimentionsを計算して、空の配列を生成
    x_dim = round((bounds[1]-bounds[0])/spacing[0]) + 1
    y_dim = round((bounds[3]-bounds[2])/spacing[1]) + 1
    z_dim = round((bounds[5]-bounds[4])/spacing[2]) + 1
    dimentions = (x_dim, y_dim, z_dim)
    #print("dimentions:", dimentions)
    img_array = np.zeros(dimentions)

    #polyデータの座標を取得して、近くの座標に対応する配列を1にする
    for i in range(num):
        point = np.array(polydata.GetPoint(i))
        point_list.append(point)
        index = []
        index.append(round((point[0]-bounds[0])/spacing[0]))
        index.append(round((point[1]-bounds[2])/spacing[1]))
        index.append(round((point[2]-bounds[4])/spacing[2]))
        img_array[index[0], index[1], index[2]] = 1
    
    if type(spacing) != list:
        spacing = list(spacing)

    #print(img_array.shape)
    imagedata = numpy_to_vtk(img_array, spacing, origin)
    return(point_list, imagedata)


def icp(source_vtp, target_vtp, max_iteration):
    # ============ run ICP ==============
    icp = vtk.vtkIterativeClosestPointTransform()
    #入力、出力データをセット
    icp.SetSource(source_vtp)
    icp.SetTarget(target_vtp)
    #剛体に設定
    #icp.CheckMeanDistanceOn()                       #
    icp.GetLandmarkTransform().SetModeToRigidBody()
    icp.DebugOn()                                   
    #icp.SetMaximumMeanDistance(0.001)               #更新前と更新後の距離の閾値

    icp.SetMaximumNumberOfIterations(max_iteration)
    icp.StartByMatchingCentroidsOff()     #重心位置合わせOFF
    icp.Modified()
    icp.Update()

    #icpの結果から4x4行列を出力
    icpTransformFilter = vtk.vtkTransformPolyDataFilter()
    icpTransformFilter.SetInputData(target_vtp)
    icpTransformFilter.SetTransform(icp)
    icpTransformFilter.Update()

    transform_matrix = icpTransformFilter.GetTransform().GetMatrix()
    result_matrix = np.identity(4)
    for i in range(4):
        for j in range(4):
            result_matrix[i][j] = transform_matrix.GetElement(i, j)

    return result_matrix
    #変換行列を返す

def pcd_mtx(pcd_array, mtx):
    paded_array = np.pad(pcd_array,((0,0),(0,1)),  mode="constant", constant_values=1)
    moved_list = []
    for i in range(paded_array.shape[0]):
        origin_point = paded_array[i, :].T
        moved_point = np.dot(mtx, origin_point)
        moved_list.append(moved_point)
        #print(origin_point, moved_point)
    moved_array = np.asarray(moved_list)[:, :3]
    return moved_array



if __name__ == '__main__':
    """ターゲット側は動かさなくてよい！！！"""
    target_path = r"D:\takahashi_k\database\us\divide\divide(1_8)"
    source_path =  r"D:\takahashi_k\database\us\divide\divide(1_8)_move(1-30)"
    out_path = r"D:\takahashi_k\database\us\divide\divide(1_8)_move(1-30)\icp"
    # source_name = "IM_0328-Annotation_1-5_5[4.548898871164251, 0.8050446465724643, 1.9129616245355314]"
    # target_name = "IM_0328-Annotation_2-5_5[4.047190233423951, 2.9217633166808348, 0.28905074948181286]"
    # source_root = os.path.join(path + "\\" + source_name + ".vtp")
    # target_root = os.path.join(path + "\\" + target_name + ".vtp")
    max_iteration = 10000

    target_rootlist = glob.glob(f'{target_path}/*1-*.vtp')
    for target_root in target_rootlist:
        target_vtp = vtp_data_loader(target_root)
        #print(target_root)
        #対応するソースのvtpを見つける
        target_name = target_root.replace(target_path, "")
        target_name = target_name.replace(".vtp", "")
        target_name = target_name.replace("\\", "")
        #print(target_name) 
        name = target_name.replace("_1", "_2")
        name = name + "_"
        #print(f"name:{name}")
        source_rootlist = glob.glob(f'{source_path}/*{name}*.vtp')

        for source_root in source_rootlist:
            source_vtp = vtp_data_loader(source_root)
            #print(source_root)
            source_name = source_root.replace(source_path, "") 
            source_name = source_name.replace(".vtp", "")
            source_name = source_name.replace("\\", "")
            #print(source_name)
            
            #ICP法によるレジストレーション
            result_matlix = icp(source_vtp, target_vtp, max_iteration)
            print(result_matlix)
            source_vtp_icped = regi_by_mtx(result_matlix, source_vtp)

            out_root = os.path.join(out_path + "\\" + source_name + "_icped_to_" + target_name + f"_{max_iteration}.vtp")
            save_vtp(source_vtp_icped, out_root)
    
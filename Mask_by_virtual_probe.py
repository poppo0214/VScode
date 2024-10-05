import numpy as np
import random
import math
import vtk
from utils.utils_vtk import vtk_data_loader, vtp_to_vti, save_vtk, save_vtp
import os
import open3d as o3d
import quaternion
import glob
import csv

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

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

def translation(Error):
    #各軸の並進移動量をランダムに決定する
    print(f"\ttranslation {Error}mm")
    x_ran = random.random()
    y_ran = random.random()
    z_ran = random.random()
    D = math.sqrt(x_ran*x_ran + y_ran*y_ran + z_ran*z_ran)
    x_error = x_ran*Error/D
    y_error = y_ran*Error/D
    z_error = z_ran*Error/D
    D = math.sqrt(x_error*x_error + y_error*y_error + z_error*z_error)
    #print(D)
    TrsMtx = np.asarray([[1,0,0,x_error],[0,1,0,y_error],[0,0,1,z_error],[0,0,0,1]])
    #print(TrsMtx)
    return TrsMtx    

def Rotation_quat(theta_deg):
    print(f"\trotation {theta_deg}deg")
    theta_rad = math.radians(theta_deg)
    vx = random.random()
    vy = random.random()
    vz = random.random()
    D = math.sqrt(vx*vx + vy*vy + vz*vz)
    vx = vx*1/D
    vy = vy*1/D
    vz = vz*1/D
    D = math.sqrt(vx*vx + vy*vy + vz*vz)
    #print(D)   
    quat = np.quaternion(math.cos(theta_rad/2), vx*math.sin(theta_rad/2), vy*math.sin(theta_rad/2), vz*math.sin(theta_rad/2))
    rot_matrix = np.asarray(quaternion.as_rotation_matrix(quat))
    #print(rot_matrix)
    #roted_pcd = pcd_data.rotate(rot_matrix)
    trs_matrix = np.pad(rot_matrix,((0,1),(0,1)),  mode="constant", constant_values=0)
    trs_matrix[3,3] = 1
    #print(trs_matrix)
    return trs_matrix

def divide_by_mask(BVN_array, Masktree, spa):
    #print(spa)
    d = (spa[0]*spa[0] + spa[1]*spa[1] + spa[2]*spa[2]) ** (0.5)
    #rint(d)
    masked_pointList =[]
    for num in range(BVN_array.shape[0]):
        point = BVN_array[num, :]
        [k, _, _] = Masktree.search_radius_vector_3d(point, d)
        if k > 0:
            masked_pointList.append(point)
            #print(point)
    return masked_pointList

def list_to_vtp(point_list):
    points = vtk.vtkPoints()                   
    vertices = vtk.vtkCellArray()

    for i in range(len(point_list)):
        point = points.InsertNextPoint(point_list[i])
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point)
        
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetVerts(vertices)
    return polydata

def output_pcd(array, root):
    print(root)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(array)    
    o3d.io.write_point_cloud(root, pcd)

def output_vtp_vti(array, vtproot, vtiroot, spa):
    print(vtproot)
    print(vtiroot)
    vtp = list_to_vtp(array)
    vti = vtp_to_vti(vtp, "manual", vtiroot, spa)
    save_vtp(vtp, vtproot)
    save_vtk(vti, vtiroot)

def give_initial_position(array, Initial_position):
    trsmtx = translation(Initial_position[0])
    rotmtx = Rotation_quat(Initial_position[1])
    mtx = np.dot(trsmtx, rotmtx)
    #mtx = np.dot(rotmtx, trsmtx)
    moved_array = pcd_mtx(array, mtx)
    #print(f"total\n{mtx}")
    return moved_array, mtx



def main(originMask_path, originMask_name, expandedBVN_path, expandedBVN_name, data_num, translation_range, rotation_range, Initial_position):
    """
    仮想プローブの撮像範囲によってマスクされた分割点群の生成
    step1. 仮想プローブを指定の並進移動（0, 101, 10），回転移動(0, 91, 10)で与える
    step2. マスクされた分割血管網の総点数が，元の拡張済み血管網の総点数の1/3以下のものは除く
    step3. 初期誤差（並進10mm, 回転10°）をランダムで与え，ソース点群を作成する
    step4. マスク，分割血管網，ソース血管網をvti, vtp, pcd形式で出力する
    step5. 分割血管網の数が30になるまで同様の処理を続ける
    """
    originMask_root = os.path.join(originMask_path + "\\" + originMask_name + ".pcd")
    expandedBVN_root = os.path.join(expandedBVN_path + "\\" + expandedBVN_name + ".pcd")
    expandedBVN_vtiroot = os.path.join(expandedBVN_path + "\\" + expandedBVN_name + ".vti")
    movedMask_path = os.path.join(expandedBVN_path + "\\movedMask")
    dividedBVN_path = os.path.join(expandedBVN_path + "\\dividedBVN")
    sourceBVN_path = os.path.join(expandedBVN_path + "\\source")

    #make_dir(movedMask_path)
    make_dir(dividedBVN_path)
    make_dir(sourceBVN_path)

    #step1. 仮想プローブを指定の並進移動（0, 101, 10），回転移動(0, 91, 10)で与える
    originMask_pcd = o3d.io.read_point_cloud(originMask_root)
    originMask_array = np.asarray(originMask_pcd.points)

    expandedBVN_pcd = o3d.io.read_point_cloud(expandedBVN_root)
    #expandedBVN_tree = o3d.geometry.KDTreeFlann(expandedBVN_pcd)
    expandedBVN_array = np.asarray(expandedBVN_pcd.points)

    __, spa, ori = vtk_data_loader(expandedBVN_vtiroot)
    originMask_array = originMask_array - np.asarray(ori)
    expandedBVN_array = expandedBVN_array - np.asarray(ori)

    
    for Error in translation_range:
        for theta in rotation_range:
            outmask_path = os.path.join(movedMask_path + f"\\{Error}mm-{theta}deg")
            outdividedBVN_path = os.path.join(dividedBVN_path + f"\\{Error}mm-{theta}deg")
            outsourceBVN_path = os.path.join(sourceBVN_path + f"\\{Error}mm-{theta}deg")
            outMTX_path = os.path.join(outsourceBVN_path + "\\TrsMtx_GT")
            # make_dir(outmask_path)
            make_dir(outdividedBVN_path)
            make_dir(outsourceBVN_path)
            make_dir(outMTX_path)
            count = 0
            countF = 0
            
            while(count<data_num):
                print(f"progress>> {count+1}/{data_num}")
                outmask_root = os.path.join(outmask_path + f"\\{expandedBVN_name}_{Error}-{theta}")
                outdividedBVN_root = os.path.join(outdividedBVN_path + f"\\{expandedBVN_name}{Error}-{theta}")
                outsourceBVN_root = os.path.join(outsourceBVN_path + f"\\{expandedBVN_name}_{Error}-{theta}")
                outMTXroot = os.path.join(outMTX_path + f"\\{expandedBVN_name}_{Error}-{theta}")

                print("move mask")
                Trs_mtx = translation(Error)
                Rot_mtx = Rotation_quat(theta)
                mtx = np.dot(Trs_mtx, Rot_mtx)
                #mtx = np.dot(Rot_mtx, Trs_mtx)
                #print(f"total\n{mtx}")
                movedMask_array = pcd_mtx(originMask_array, mtx)
                movedMask_pcd = o3d.geometry.PointCloud()
                movedMask_pcd.points = o3d.utility.Vector3dVector(movedMask_array)
                #output_pcd(movedMask_array, os.path.join(outmask_root + ".pcd"))  
                
                movedMask_tree = o3d.geometry.KDTreeFlann(movedMask_pcd)
                dividedBVN_array = divide_by_mask(expandedBVN_array, movedMask_tree, spa)
                #dividedBVN_array = divide_by_mask(movedMask_array, expandedBVN_tree, spa)

                #step2. マスクされた分割血管網の総点数が，元の拡張済み血管網の総点数の1/10以下のものは除く
                print(f"number of points>> {len(dividedBVN_array)}/{len(expandedBVN_array)}")
                if len(dividedBVN_array) == 0:
                    print("None of the point clouds were cut out.")
                elif len(dividedBVN_array) > (len(expandedBVN_array)/10):
                    num = str(count).zfill(3)

                    # step3. 初期誤差（並進10mm, 回転10°）をランダムで与え，ソース点群を作成する
                    print("move BVN")
                    movedBVN_array, trs_mtx = give_initial_position(dividedBVN_array, Initial_position)

                    # step4. マスク,分割血管網，ソース血管網をそれぞれvti, vtp, pcd形式で出力する
                    movedMask_array = (np.asarray(movedMask_array) + np.asarray(ori)).tolist()
                    dividedBVN_array = (np.asarray(dividedBVN_array) + np.asarray(ori)).tolist()
                    movedBVN_array = (np.asarray(movedBVN_array) + np.asarray(ori)).tolist()
                    #output_pcd(movedMask_array, os.path.join(outmask_root + "-{num}.pcd"))
                    output_pcd(dividedBVN_array, os.path.join(outdividedBVN_root + f"-{num}.pcd"))
                    output_pcd(movedBVN_array, os.path.join(outsourceBVN_root + f"-{num}.pcd"))

                    np.savetxt(os.path.join(outMTXroot + f"-{num}.tsv"), trs_mtx, delimiter='\t')
                    #output_vtp_vti(movedMask_array, os.path.join(outmask_root + "-{num}.vtp"), os.path.join(outmask_root + "-{num}.vti"), spa)
                    #output_vtp_vti(dividedBVN_array, os.path.join(outdividedBVN_root + "-{num}.vtp"), os.path.join(outdividedBVN_root + "-{num}.vti"), spa)
                    #output_vtp_vti(movedBVN_array, os.path.join(outsourceBVN_root + f"-{num}.vtp"), os.path.join(outsourceBVN_root + f"-{num}.vti"), spa)
                    
                    count += 1

                else:
                    numF = str(countF).zfill(3)
                    # step3. 初期誤差（並進10mm, 回転10°）をランダムで与え，ソース点群を作成する
                    print("move BVN")
                    movedBVN_array, trs_mtx = give_initial_position(dividedBVN_array, Initial_position)

                    # step4. マスク,分割血管網，ソース血管網をそれぞれvti, vtp, pcd形式で出力する
                    movedMask_array = (np.asarray(movedMask_array) + np.asarray(ori)).tolist()
                    dividedBVN_array = (np.asarray(dividedBVN_array) + np.asarray(ori)).tolist()
                    movedBVN_array = (np.asarray(movedBVN_array) + np.asarray(ori)).tolist()
                    #output_pcd(movedMask_array, os.path.join(outmask_root + f"-Few{numF}.pcd"))
                    output_pcd(dividedBVN_array, os.path.join(outdividedBVN_root + f"-Few{numF}.pcd"))
                    output_pcd(movedBVN_array, os.path.join(outsourceBVN_root + f"-Few{numF}.pcd"))

                    np.savetxt(os.path.join(outMTXroot + f"-Few{numF}.tsv"), trs_mtx, delimiter='\t')
                    #output_vtp_vti(movedMask_array, os.path.join(outmask_root + f"-Few{numF}.vtp"), os.path.join(outmask_root + "-Few{numF}.vti"), spa)
                    #output_vtp_vti(dividedBVN_array, os.path.join(outdividedBVN_root + "-Few{numF}.vtp"), os.path.join(outdividedBVN_root + "-Few{numF}.vti"), spa)
                    #output_vtp_vti(movedBVN_array, os.path.join(outsourceBVN_root + f"-Few{numF}.vtp"), os.path.join(outsourceBVN_root + f"-Few{numF}.vti"), spa)
                    
                    
                    countF += 1



if __name__ == '__main__':
    """
    originMask:基準となるマスク
    expandedBVN:拡張済み血管網
    """
    originMask_path = r"D:\takahashi_k\registration(expandedBVN)\makino"
    originMask_name = "makino_mask"
    expandedBVN_path = r"D:\takahashi_k\registration(expandedBVN)\makino"
    expandedBVN_name = "makino"
    data_num = 30
    translation_range = range(0, 1, 10)
    rotation_range = range(20, 51, 10)
    Initial_position = [10, 10]             #初期位置の誤差[並進誤差(mm)，回転誤差(°)]

    main(originMask_path, originMask_name, expandedBVN_path, expandedBVN_name, data_num, translation_range, rotation_range, Initial_position)
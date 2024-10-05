import os
import numpy as np
import glob
import math
import open3d as o3d
import quaternion
from scipy.spatial.transform import Rotation
from scipy import linalg

"""target = 正解のとき"""
"""angular error, translation error"""
"""https://programming-surgeon.com/script/euler-python-script/"""
"""https://qiita.com/eduidl/items/a6e55101c6fe15bf7b8f"""


def calculate_FRE(regied_array, target_array):
    if regied_array.shape != target_array.shape:
        print("Error: shape of regied array have different shape of target array!!!")
        return 0
    else:
        error = 0
        for i in range(regied_array.shape[0]):
            regied_point = regied_array[i, :]
            target_point = target_array[i, :]
            x_error = target_point[0] - regied_point[0]
            y_error = target_point[1] - regied_point[1]
            z_error = target_point[2] - regied_point[2]
            error += math.sqrt(x_error*x_error + y_error*y_error + z_error*z_error)
        return (error/regied_array.shape[0])

def calculate_EulerAngle(matrix):
    """
    入力
        matrix = 3x3回転行列
        oreder = 回転順　たとえば X, Z, Y順なら'xzy'
    出力
        theta1, theta2, theta3 = 回転角度 回転順にtheta 1, 2, 3
    """
    r11, r12, r13 = matrix[0]
    r21, r22, r23 = matrix[1]
    r31, r32, r33 = matrix[2]

    theta1 = np.arctan(-r23 / r33)
    theta2 = np.arctan(r13 * np.cos(theta1) / r33)
    theta3 = np.arctan(-r12 / r11)

    theta1 = theta1 * 180 / np.pi
    theta2 = theta2 * 180 / np.pi
    theta3 = theta3 * 180 / np.pi
    if theta1 > 180:
        theta1 = -360 + theta1
    if theta2 > 180:
        theta2 = -360 + theta2
    if theta3 > 180:
        theta3 = -360 + theta3
    return (theta1, theta2, theta3)

def calculate_quaternion(matrix):
    quat = quaternion.from_rotation_matrix(matrix, nonorthogonal=True)
    theta = quat.angle()*180/np.pi
    if theta > 180:
        theta = -360 + theta
    return(theta)

def calculate_rot_error(RotMtx_GT, RotMtx_EM):
    error_rot = Rotation.from_matrix(RotMtx_EM @ RotMtx_GT.T)
    return linalg.norm(error_rot.as_rotvec())

# def calculate_RRE(RotMtx_GT, RotMtx_EM):
#     print(RotMtx_GT, RotMtx_EM)
#     print("mid-calculation", math.acos(np.matrix.trace(RotMtx_EM @ RotMtx_GT.T-1))/2)
#     RRE = math.acos(np.matrix.trace(RotMtx_EM @ RotMtx_GT.T-1)/2)
#     return RRE

if __name__ == '__main__':
    target_path = r"D:\takahashi_k\registration\Y(filled)"
    #path = r"D:\takahashi_k\registration\vessel(filled)\translation_10mm"

    for num in range(180, 360, 30):
    #for num in range(12, 13, 1):
        path = os.path.join(target_path + f"\\rotation_{num}deg")
        #path = os.path.join(target_path + f"\\translation_{num}mm")
        outtsv_root = os.path.join(target_path + f"\\evaluation.tsv")
        regied_path = os.path.join(path + f"\\icp")
        TrsMtx_GT_path = os.path.join(path + f"\\TrsMtx_GT")
        TrsMtx_EM_path = os.path.join(regied_path + f"\\TrsMtx_EM")
        tsv_root = os.path.join(regied_path + f"\\evaluation.tsv")
        ftsv = open(tsv_root, "a")
        ftsv.write(f"target_name\tsource_name\tRRE(rad)\tRRE(deg)\tRTE\tFRE\n")
        Give_rot = True

        rootlist = glob.glob(f'{regied_path}/*.pcd')
        for regied_root in rootlist:
            #print("regi: ", regied_root)
            regied_pcd= o3d.io.read_point_cloud(regied_root)
            regied_array = np.asarray(regied_pcd.points)
            filename = regied_root.replace(regied_path, "")
            filename = filename.replace(".pcd", "")
            target_name = filename.split("_")[-1]
            source_name = filename.split("_")[0] + "_" + filename.split("_")[1]
            print(target_name, "\n", source_name)

            target_root = os.path.join(target_path + "\\" + target_name + ".pcd")
            #print("target: ", target_root)
            target_pcd= o3d.io.read_point_cloud(target_root)
            target_array = np.asarray(target_pcd.points)

            if Give_rot:
                TrsMtx_GT_root = os.path.join(TrsMtx_GT_path + "\\" + source_name + ".tsv")
                print("TrsMtx_GT_root: ", TrsMtx_GT_root)
                fgt = open(TrsMtx_GT_root, 'r')
                TrsMtx_GT = fgt.read()
                TrsMtx_GT = TrsMtx_GT.replace(",", "")
                TrsMtx_GT = TrsMtx_GT.split()
                TrsMtx_GT = np.array(TrsMtx_GT, dtype=float)
                TrsMtx_GT = TrsMtx_GT.reshape(4,4)
                fgt.close()

            else:
                TrsMtx_GT = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

            TrsMtx_EM_root = os.path.join(TrsMtx_EM_path + filename + ".tsv")
            print("TrsMtx_EM_root", TrsMtx_EM_root)
            fem = open(TrsMtx_EM_root, 'r')
            TrsMtx_EM = fem.read()
            TrsMtx_EM = TrsMtx_EM.replace(",", "")
            TrsMtx_EM = TrsMtx_EM.split()
            TrsMtx_EM = np.array(TrsMtx_EM, dtype=float)
            TrsMtx_EM = TrsMtx_EM.reshape(4,4)
            fem.close()

           
            RotMtx_GT = np.linalg.inv(TrsMtx_GT[:3, :3])
            RotMtx_EM = TrsMtx_EM[:3, :3]
            x_degree, y_degree, z_degree = calculate_EulerAngle(RotMtx_EM)
            print(f"degree error [x, y, z]: [{x_degree}, {y_degree}, {z_degree}]")
            quat_degree = calculate_quaternion(RotMtx_EM)
            print(f"quat degree error: {quat_degree}")
            rot_error = calculate_rot_error(RotMtx_GT, RotMtx_EM)
            # RRE = calculate_RRE(RotMtx_GT, RotMtx_EM)
            # print(f"rot error: {rot_error}\nRRE: {RRE}")

            FRE = calculate_FRE(regied_array, target_array)
            print(f"FRE: {FRE}")
        #     ftsv.write(f"{target_name}\t{source_name}\t{FRE}\t{rot_error}\t{quat_degree}\t{x_degree}\t{y_degree}\t{z_degree}\n")

        # ftsv.close()



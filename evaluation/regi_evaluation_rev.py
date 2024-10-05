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


def calculate_RRE_RTE(TrsMtx_GT, TrsMtx_EM):
    # print(TrsMtx_GT)
    # print(np.dot(TrsMtx_GT, np.linalg.inv(TrsMtx_GT)))
    # print(TrsMtx_EM)
    # print(np.dot(TrsMtx_EM, np.linalg.inv(TrsMtx_EM)))
    deltaT = np.dot(TrsMtx_GT, np.linalg.inv(TrsMtx_EM))
    #print("deltaT")
    #print(deltaT)
    deltaR = deltaT[:3, :3]
    #print(deltaR)
    deltat = np.ravel(deltaT[:3, 3:4])
    #print(deltat)
    #print((-1+np.matrix.trace(deltaR))/2)
    Er = math.acos(np.clip((-1+np.matrix.trace(deltaR))/2, -1.0, 1.0))
    #print(Er)
    Et = np.linalg.norm(deltat)
    #print(Et)
    return Er, Et

if __name__ == '__main__':
    #target_path = r"D:\takahashi_k\registration\Y(surface)"
    #path = r"D:\takahashi_k\registration\vessel(filled)\translation_10mm"
    # data_name = ["tube(filled)", "tube(surface)", "Y(filled)", "Y(surface)", "vessel(filled)", "vessel(surface)"]
    path = r"D:\takahashi_k\registration(expandedBVN)\makino"
    Give_rot = True
    alltsv_root = os.path.join(path + f"\\evaluation_all.tsv")
    alltsv = open(alltsv_root, "a")
    alltsv.write(f"name\tRRE[rad]\tRRE[deg]\tRTE[mm]\tFRE[mm]\n")

    for error in range(0, 81, 10):
        for theta in range(0, 91, 10):
            num = 0
            sum_RRErad = 0
            sum_RREdeg = 0
            sum_RTE = 0
            sum_FRE = 0
            #source_path =  os.path.join(path + f"\\source\\{error}mm-{theta}deg")
    
            #path = os.path.join(target_path + f"\\rotation_{num}deg")
            inpath = os.path.join(path + f"\\source\\{error}mm-{theta}deg")
            #outtsv_root = os.path.join(target_path + f"\\evaluation_ransac.tsv")
            regied_path = os.path.join(inpath + f"\\FPFH_RANSAC_icp")
            #regied_path = os.path.join(path + f"\\icp")
            TrsMtx_GT_path = os.path.join(inpath + f"\\TrsMtx_GT")
            TrsMtx_EM_path = os.path.join(regied_path + f"\\TrsMtx_EM")
            #TrsMtx_EM_path = os.path.join(regied_path + f"\\RansacMtx_EM")
            # tsv_root = os.path.join(regied_path + f"\\evaluation_total.tsv")
            tsv_root = os.path.join(regied_path + f"\\evaluation.tsv")
            ftsv = open(tsv_root, "a")
            ftsv.write(f"name\tRRE[rad]\tRRE[deg]\tRTE[mm]\tFRE[mm]\n")

            rootlist = glob.glob(f'{regied_path}/*.pcd')
            for regied_root in rootlist:                
                print("regi: ", regied_root)
                regied_pcd= o3d.io.read_point_cloud(regied_root)
                regied_array = np.asarray(regied_pcd.points)   
             
                truth_root = regied_root.replace("source", "dividedBVN")
                truth_root = truth_root.replace("\\FPFH_RANSAC_icp", "")
                filename = regied_root.replace(regied_path, "")
                filename = filename.replace(".pcd", "")
                sourcename = filename.split("_Regied")[0]
                truth_root = truth_root.replace(filename, sourcename)
                print("truth: ",truth_root)
                truth_pcd = o3d.io.read_point_cloud(truth_root)
                truth_array = np.asarray(truth_pcd.points)
                
                
                """RANSACの場合"""
                if Give_rot:
                    TrsMtx_GT_root = os.path.join(TrsMtx_GT_path + "\\" + sourcename + ".tsv")
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

                #print(TrsMtx_GT)
                TrsMtx_GT = np.linalg.inv(TrsMtx_GT)
                #print(TrsMtx_GT)
                TrsMtx_EM_root = os.path.join(TrsMtx_EM_path + filename + ".tsv")
                #print("TrsMtx_EM_root", TrsMtx_EM_root)
                fem = open(TrsMtx_EM_root, 'r')
                TrsMtx_EM = fem.read()
                TrsMtx_EM = TrsMtx_EM.replace(",", "")
                TrsMtx_EM = TrsMtx_EM.split()
                TrsMtx_EM = np.array(TrsMtx_EM, dtype=float)
                TrsMtx_EM = TrsMtx_EM.reshape(4,4)
                fem.close()

                #print(TrsMtx_EM)

                RRE, RTE = calculate_RRE_RTE(TrsMtx_GT, TrsMtx_EM)
                # RotMtx_GT = TrsMtx_GT[:3, :3]
                # RotMtx_EM = TrsMtx_EM[:3, :3]
                FRE = calculate_FRE(regied_array, truth_array)
                print(f"RRE[rad]/RRE[deg]/RTE[mm]/FRE[mm]: {RRE}/{RRE*180/np.pi}/{RTE}/{FRE}")
                ftsv.write(f"{filename}\t{RRE}\t{RRE*180/np.pi}\t{RTE}\t{FRE}\n")

                if not "Few" in regied_path:
                    num +=1
                    sum_RRErad += RRE
                    sum_RREdeg += RRE*180/np.pi
                    sum_RTE += RTE
                    sum_FRE += FRE

            ftsv.close()
            ave_RRErad = sum_RRErad/num
            ave_RREdeg = sum_RREdeg/num
            ave_RTE = sum_RTE/num
            ave_FRE = sum_FRE/num
            alltsv.write(f"{error}mm-{theta}deg\t{ave_RRErad}\t{ave_RREdeg}\t{ave_RTE}\t{ave_FRE}\n")

    alltsv.close()




# if __name__ == '__main__':
#     #target_path = r"D:\takahashi_k\registration\Y(surface)"
#     #path = r"D:\takahashi_k\registration\vessel(filled)\translation_10mm"
#     # data_name = ["tube(filled)", "tube(surface)", "Y(filled)", "Y(surface)", "vessel(filled)", "vessel(surface)"]
#     data_name = ["tube(filled-2)", "Y(filled-2)", "vessel(filled-2)"]
    
#     for data in data_name:
#         #for num in range(0, 360, 30):
#         for num in range(0, 16, 1):
#             target_path = os.path.join(f"D:\\takahashi_k\\registration(model)\\" + data)
#             #path = os.path.join(target_path + f"\\rotation_{num}deg")
#             path = os.path.join(target_path + f"\\translation_{num}mm")
#             #outtsv_root = os.path.join(target_path + f"\\evaluation_ransac.tsv")
#             # regied_path = os.path.join(path + f"\\FPFH_RANSAC_icp")
#             regied_path = os.path.join(path + f"\\icp")
#             TrsMtx_GT_path = os.path.join(path + f"\\TrsMtx_GT")
#             TrsMtx_EM_path = os.path.join(regied_path + f"\\TrsMtx_EM")
#             #TrsMtx_EM_path = os.path.join(regied_path + f"\\RansacMtx_EM")
#             # tsv_root = os.path.join(regied_path + f"\\evaluation_total.tsv")
#             tsv_root = os.path.join(regied_path + f"\\evaluation.tsv")
#             ftsv = open(tsv_root, "a")
#             ftsv.write(f"target_name\tsource_name\tRRE(rad)\tRRE(deg)\tRTE\tFRE\n")
#             Give_rot = True

#             rootlist = glob.glob(f'{regied_path}/*.pcd')
#             for regied_root in rootlist:
#                 #print("regi: ", regied_root)
#                 regied_pcd= o3d.io.read_point_cloud(regied_root)
#                 regied_array = np.asarray(regied_pcd.points)
#                 filename = regied_root.replace(regied_path, "")
#                 filename = filename.replace(".pcd", "")
#                 """RANSACの場合"""
#                 # filename = filename.replace("Regied", "RANSAC")
#                 print(filename)

#                 target_name = filename.split("_")[-1]
#                 source_name = filename.split("_")[0] + "_" + filename.split("_")[1]              
#                 print(target_name, "\n", source_name)

#                 target_root = os.path.join(target_path + "\\" + target_name + ".pcd")
#                 #print("target: ", target_root)
#                 target_pcd= o3d.io.read_point_cloud(target_root)
#                 target_array = np.asarray(target_pcd.points)

#                 if Give_rot:
#                     TrsMtx_GT_root = os.path.join(TrsMtx_GT_path + "\\" + source_name + ".tsv")
#                     print("TrsMtx_GT_root: ", TrsMtx_GT_root)
#                     fgt = open(TrsMtx_GT_root, 'r')
#                     TrsMtx_GT = fgt.read()
#                     TrsMtx_GT = TrsMtx_GT.replace(",", "")
#                     TrsMtx_GT = TrsMtx_GT.split()
#                     TrsMtx_GT = np.array(TrsMtx_GT, dtype=float)
#                     TrsMtx_GT = TrsMtx_GT.reshape(4,4)
#                     fgt.close()

#                 else:
#                     TrsMtx_GT = np.asarray([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

#                 #print(TrsMtx_GT)
#                 TrsMtx_GT = np.linalg.inv(TrsMtx_GT)
#                 #print(TrsMtx_GT)
#                 TrsMtx_EM_root = os.path.join(TrsMtx_EM_path + filename + ".tsv")
#                 #print("TrsMtx_EM_root", TrsMtx_EM_root)
#                 fem = open(TrsMtx_EM_root, 'r')
#                 TrsMtx_EM = fem.read()
#                 TrsMtx_EM = TrsMtx_EM.replace(",", "")
#                 TrsMtx_EM = TrsMtx_EM.split()
#                 TrsMtx_EM = np.array(TrsMtx_EM, dtype=float)
#                 TrsMtx_EM = TrsMtx_EM.reshape(4,4)
#                 fem.close()

#                 #print(TrsMtx_EM)

#                 RRE, RTE = calculate_RRE_RTE(TrsMtx_GT, TrsMtx_EM)
#                 # RotMtx_GT = TrsMtx_GT[:3, :3]
#                 # RotMtx_EM = TrsMtx_EM[:3, :3]
#                 FRE = calculate_FRE(regied_array, target_array)
#                 print(f"RRE(rad)/RRE(deg)/RTE/FRE: {RRE}/{RRE*180/np.pi}/{RTE}/{FRE}")
#                 ftsv.write(f"{target_name}\t{source_name}\t{RRE}\t{RRE*180/np.pi}\t{RTE}\t{FRE}\n")

#             ftsv.close()



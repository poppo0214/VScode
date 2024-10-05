import os
import sys
import numpy as np
#from utils.utils_vtk import vtp_data_loader, save_vtp
import glob
import csv
import pprint
import math
sys.path.append(os.pardir)
from utils.utils_vtk import vtp_data_loader, save_vtp

if __name__ == '__main__':
    correct_path = r"D:\takahashi_k\database\us\divide\divide(1_8)"
    regied_path = r"D:\takahashi_k\database\us\divide\divide(1_8)_move(1-30)\icp"
    csvroot = os.path.join(regied_path + "\\MSE.tsv")
    fcsv = open(csvroot, "a")
    fcsv.write("correct(before registrarion)\tsource(registrarioned)\ttarget\titration num\tMSE\n")
    
    #file_name = "IM_0328-Annotation"
    #inroot = os.path.join(Inpath + "\\" + file_name + ".vtp")

    rootlist = glob.glob(f'{correct_path}/*2-*.vtp')
    for correct_root in rootlist:
        #vtpデータ読み込み
        #print(correct_root)
        correct_poly = vtp_data_loader(correct_root)
        correct_name = correct_root.replace(correct_path, "")
        correct_name = correct_name.replace(".vtp", "")
        correct_name = correct_name.replace("\\", "")
        print(correct_name)

        regied_rootlist = glob.glob(f'{regied_path}/*{correct_name}_*.vtp')
        print(regied_rootlist)
        for regied_root in regied_rootlist:
            regied_poly = vtp_data_loader(regied_root)
            regied_name = regied_root.replace(regied_path, "")
            """名前取得（icp時の設定）"""
            name = regied_name.split("_icped_to_")
            source_name = str(name[0])
            target_number = name[1].replace(".vtp", "")
            temp = target_number
            temp_list = temp.split("_")
            iteration_num = temp_list[-1]
            target_name = target_number.replace(iteration_num, "")

            #print(source_name, target_name, iteration_num)

            num = correct_poly.GetNumberOfPoints()            #vtpのポイント数を取得
            error = 0
            for i in range(num):
                correct_point = np.array(correct_poly.GetPoint(i))
                regied_point = np.array(regied_poly.GetPoint(i))
                x_error = correct_point[0] - regied_point[0]
                y_error = correct_point[1] - regied_point[1]
                z_error = correct_point[2] - regied_point[2]
                error += math.sqrt(x_error*x_error + y_error*y_error + z_error*z_error)
            #print(error)
            MSE = error/num
            print(MSE)
            fcsv.write(f"{correct_name}\t{source_name}\t{target_name}\t{iteration_num}\t{MSE}\n")
    fcsv.close()



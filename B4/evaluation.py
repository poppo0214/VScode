import numpy as np
import itk
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import cv2
import glob

"""2値化すること!!!!!!!!!"""   
filtered_path = r'D:\takahashi_k\PaperData\AS_or_masked_VP(500)_closing(5)_VP'
txt_name = "AS_or_masked_VP(500)_closing(5)_"

if __name__ == '__main__': 
    segmented_path = r'd:\takahashi_k\database\us\annotation_all'
    vth_list = [100000]   
    for vth in vth_list:
        In_path = os.path.join(filtered_path + f"\\AS_or_masked_VP(500)_closing(5)_VP({vth})")
        Out_path = In_path
        out_root = os.path.join(Out_path + "\\" + txt_name + f"({vth})_evaluation.tsv")
    # #bnth_list = [10, 20, 30, 40, 50, 60]
    # for bnth in bnth_list:
    #     In_path = os.path.join(filtered_path + f"\\bn({bnth})")        
    #     Out_path = In_path
    #     out_root = os.path.join(Out_path + "\\" + txt_name + f"_bn({bnth})_evaluation.tsv")
    # ksize_list = [5,10,15]
    # for ksize in ksize_list:
    #     In_path = os.path.join(filtered_path + f"\\AS_frg_VP(500)_closing({ksize})")        
    #     Out_path = In_path
    #     out_root = os.path.join(Out_path + "\\" + txt_name + f"({ksize})_evaluation.tsv")
    
    
    
    # In_path = filtered_path
    # Out_path = In_path
    # out_root = os.path.join(Out_path + "\\" + txt_name + f"_evaluation.tsv")

        f = open(out_root, 'a')
        f.write("filename\tAccuracy\tpresision\tRecall\tDice\tIoU\n")

        rootlist = glob.glob(f'{In_path}/*.vti')
        for root in rootlist:
            filtered_array, spa, ori = vtk_data_loader(root)
            filename = root.replace(In_path, "")
            filename = filename.replace(".vti", "")
            num = filename.find('Origin')
            personname = filename[0:num]

            print(personname, num)

            segmented_root = os.path.join(segmented_path + personname + "seg.vti")
            segmented_array, spa, ori = vtk_data_loader(segmented_root)
            print(segmented_root)


            
            array1 = filtered_array
            print(array1.shape, np.max(array1), np.min(array1))
            array2 = segmented_array
            print(array2.shape, np.max(array2), np.min(array2))
            
            N = array1.size
            TP_FN = np.sum(array2)
            TP_FP = np.sum(array1)

            mult = array1*array2
            TP = np.sum(np.abs(mult)) 
            FN = TP_FN - TP
            FP = TP_FP - TP
            TN = N - (TP+FN+FP)

            Accuracy = (TP + TN)/N          #ボリューム全体のうち、どれだけ正しく血管と背景を区別できたか
            Presision = TP/TP_FP            #抽出された血管のうち、実際の血管の割合（誤検知NG）
            Recall = TP/TP_FN               #実際の血管のうち、抽出された血管の割合（見逃しNG）
            Dice = TP/(TP+(FP+FN)/2)        #PresisionとRecallの調和平均
            IoU = TP/(TP_FP+FN)             #実際の血管と抽出した血管を足した領域に占めるTPの割合  

            print((f"{filename}\t{Accuracy}\t{Presision}\t{Recall}\t{Dice}\t{IoU}\n"))
            f.write(f"{filename}\t{Accuracy}\t{Presision}\t{Recall}\t{Dice}\t{IoU}\n")
        f.close
                



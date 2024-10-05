import os
import numpy as np
from utils.utils_vtk import vtp_data_loader, save_vtp, vtp_to_vti, save_vtk
import glob
import vtk


if __name__ == '__main__':
    target_path = r"D:\takahashi_k\database\us\divide\divide(1_8)"
    regied_path = r"D:\takahashi_k\database\us\divide\divide(1_8)_move(1-30)\icp"
    out_path = r"D:\takahashi_k\database\us\divide\divide(1_8)_move(1-30)\icp\expand"
    reference_path = r"D:\takahashi_k\database\us\divide"

    rootlist = glob.glob(f'{target_path}/*1-*.vtp')
    for target_root in rootlist:
        #vtpデータ読み込み
        #print(correct_root)
        target_poly = vtp_data_loader(target_root)
        target_name = target_root.replace(target_path, "")
        target_name = target_name.replace(".vtp", "")
        target_name = target_name.replace("\\", "")
        print(target_name)
        name = target_name.split("_")[0]
        number = target_name.split("_")[2]
        filename = name + "_IM_" + number
        print(filename)

        regied_rootlist = glob.glob(f'{regied_path}/*{target_name}_*.vtp')
        #print(regied_rootlist)
        for regied_root in regied_rootlist:
            print(regied_root)
            regied_poly = vtp_data_loader(regied_root)
            out_name = regied_root.replace(regied_path, "")
            out_name = out_name.replace(".vtp", "")
            outvtp = os.path.join(out_path + out_name + "_expand.vtp")
            outvti = os.path.join(out_path + out_name + "_expand.vti")
        
            #vtpデータ読み込み
            regiedPoly = vtp_data_loader(regied_root)
            targetPoly = vtp_data_loader(target_root)

            regied_num = regiedPoly.GetNumberOfPoints()            #vtpのポイント数を取得
            target_num = targetPoly.GetNumberOfPoints()            #vtpのポイント数を取得
            #point_list = []
            points = vtk.vtkPoints()
            vertices = vtk.vtkCellArray()

            for i in range(regied_num):
                #regied_point = np.array(regiedPoly.GetPoint(i))
                #point_list.append(point)
                point = points.InsertNextPoint(np.array(regiedPoly.GetPoint(i)))
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(point)
                OutPoly = vtk.vtkPolyData()
                OutPoly.SetPoints(points)
                OutPoly.SetVerts(vertices)

            for j in range(target_num):
               # point = np.array(targetPoly.GetPoint(j))
                #point_list.append(point)
                point = points.InsertNextPoint(np.array(targetPoly.GetPoint(j)))
                vertices.InsertNextCell(1)
                vertices.InsertCellPoint(point)
                OutPoly = vtk.vtkPolyData()
                OutPoly.SetPoints(points)
                OutPoly.SetVerts(vertices)
            
            #save as vtp
            save_vtp(OutPoly, outvtp)
            reference_root = os.path.join(reference_path + "\\" + filename + ".vti")
            vti_data = vtp_to_vti(OutPoly, "reference", root=reference_root)
            save_vtk(vti_data, outvti)
    
        


import os
import numpy as np
from utils.utils_vtk import vtp_data_loader, save_vtp
import glob
import vtk

# """1つのデータに複数の値で分割する場合"""
# if __name__ == '__main__':
#     Inpath = r"D:\takahashi_k\database\us\devide\YAMAGUCHI_0328\move"
#     Outpath = r"D:\takahashi_k\database\us\devide\YAMAGUCHI_0328\move-divide"
#     file_name = "IM_0328-Annotation"
#     inroot = os.path.join(Inpath + "\\" + file_name + ".vtp")
#     txtroot = os.path.join(Outpath + "\\" + file_name + "_devide_memo.txt")

#     cut_ratio = 1/4         #全体の何割の点をカットするか
#     step = 10               #ステップ数

#     #vtpデータ読み込み
#     InPoly = vtp_data_loader(inroot)
#     num = InPoly.GetNumberOfPoints()            #vtpのポイント数を取得
#     point_list = []

#     f = open(txtroot, 'a')
#     f.write(f"{file_name}\t{num}\n")

#     for i in range(num):
#         point = np.array(InPoly.GetPoint(i))
#         point_list.append(point)
#     print(len(point_list))

#     #リスト生成
#     list1 = []
#     list2 = []
#     for i in range(0,step,1):
#         cut_point = round(num * cut_ratio * (i+1)/step)
#         print(cut_point)
#         temp1 = point_list[0:num-cut_point]
#         temp2 = point_list[cut_point:num]
#         list1.append(temp1)
#         list2.append(temp2)
    
#     #リストからvtpデータを作成
#     for i in range(0,step,1):
#         cut_point = round(num * cut_ratio * (i+1)/step)
#         points1 = vtk.vtkPoints()
#         points2 = vtk.vtkPoints()
#         vertices1 = vtk.vtkCellArray()
#         vertices2 = vtk.vtkCellArray()

#         for j in range(0,num-cut_point,1):
#             point1 = points1.InsertNextPoint(list1[i][j])
#             #print(point1)
#             vertices1.InsertNextCell(1)
#             vertices1.InsertCellPoint(point1)

#             point2 = points2.InsertNextPoint(list2[i][j])
#             #print(point2)
#             vertices2.InsertNextCell(1)
#             vertices2.InsertCellPoint(point2)
            
#             OutPoly1 = vtk.vtkPolyData()
#             OutPoly1.SetPoints(points1)
#             OutPoly1.SetVerts(vertices1)
#             OutPoly2 = vtk.vtkPolyData()
#             OutPoly2.SetPoints(points2)
#             OutPoly2.SetVerts(vertices2)
        
#         out_vtp1 = os.path.join(Outpath + "\\" + file_name + f"_1-{i+1}.vtp")
#         out_vtp2 = os.path.join(Outpath + "\\" + file_name + f"_2-{i+1}.vtp")
#         save_vtp(OutPoly1, out_vtp1)
#         save_vtp(OutPoly2, out_vtp2)
#         f.write(f"{file_name}_1-{i+1}\t0\t{num-cut_point}\n")
#         f.write(f"{file_name}_2-{i+1}\t{cut_point}\t{num}\n")
#         f.close


# """複数のデータに対して1つの値で分割する場合"""
# if __name__ == '__main__':
#     Inpath = r"D:\takahashi_k\database\us\divide\BUNKATUYOU"
#     Outpath = r"D:\takahashi_k\database\us\divide\BUNKATUYOU\divide"
#     cut_ratio = 1/8         #全体の何割の点をカットするか
#     txtroot = os.path.join(Outpath + "\\devide_memo.txt")
#     f = open(txtroot, 'a')

#     rootlist = glob.glob(f'{Inpath}/*.vtp')
#     for Inroot in rootlist:
#         #vtpデータ読み込み
#         InPoly = vtp_data_loader(Inroot)
#         file_name = Inroot.replace(Inpath, "")
#         file_name = file_name.replace(".vtp", "")
#         print(file_name)

#         #vtpデータ読み込み
#         InPoly = vtp_data_loader(Inroot)
#         num = InPoly.GetNumberOfPoints()            #vtpのポイント数を取得
        
#         f.write(f"{file_name}\t{num}\n")

#         point_list = []
#         for i in range(num):
#             point = np.array(InPoly.GetPoint(i))
#             point_list.append(point)
#         print(len(point_list))

#         #リスト生成
#         cut_point = round(num * cut_ratio)
#         print(cut_point)
#         temp1 = point_list[0:num-cut_point]
#         temp2 = point_list[cut_point:num]

#         points1 = vtk.vtkPoints()
#         points2 = vtk.vtkPoints()
#         vertices1 = vtk.vtkCellArray()
#         vertices2 = vtk.vtkCellArray()

#         for i in range(0,num-cut_point,1):
#             point1 = points1.InsertNextPoint(temp1[i])
#             #print(point1)
#             vertices1.InsertNextCell(1)
#             vertices1.InsertCellPoint(point1)

#             point2 = points2.InsertNextPoint(temp2[i])
#             #print(point2)
#             vertices2.InsertNextCell(1)
#             vertices2.InsertCellPoint(point2)
            
#             OutPoly1 = vtk.vtkPolyData()
#             OutPoly1.SetPoints(points1)
#             OutPoly1.SetVerts(vertices1)
#             OutPoly2 = vtk.vtkPolyData()
#             OutPoly2.SetPoints(points2)
#             OutPoly2.SetVerts(vertices2)
        
#         out_vtp1 = os.path.join(Outpath + "\\" + file_name + f"_1.vtp")
#         out_vtp2 = os.path.join(Outpath + "\\" + file_name + f"_2.vtp")
#         save_vtp(OutPoly1, out_vtp1)
#         save_vtp(OutPoly2, out_vtp2)
#         f.write(f"{file_name}_1\t0\t{num-cut_point}\n")
#         f.write(f"{file_name}_2\t{cut_point}\t{num}\n")
#     f.close


"""複数のデータに対して複数の値で分割する場合"""
if __name__ == '__main__':
    Inpath = r"D:\takahashi_k\database\us\divide"
    Outpath = r"D:\takahashi_k\database\us\divide\divide(^1_4)"
    txtroot = os.path.join(Outpath + "\\devide_memo.txt")
    f = open(txtroot, 'a')

    cut_ratio = 1/4         #全体の何割の点をカットするか
    step = 10               #ステップ数

    rootlist = glob.glob(f'{Inpath}/*.vtp')
    for Inroot in rootlist:
        #vtpデータ読み込み
        InPoly = vtp_data_loader(Inroot)
        file_name = Inroot.replace(Inpath, "")
        file_name = file_name.replace(".vtp", "")
        print(file_name)

        #vtpデータ読み込み
        InPoly = vtp_data_loader(Inroot)
        num = InPoly.GetNumberOfPoints()            #vtpのポイント数を取得        
        f.write(f"{file_name}\t{num}\n")
        
        point_list = []
        for i in range(num):
            point = np.array(InPoly.GetPoint(i))
            point_list.append(point)
        print(len(point_list))

        #リスト生成
        list1 = []
        list2 = []
        for i in range(0,step,1):
            cut_point = round(num * cut_ratio * (i+1)/step)
            print(cut_point)
            temp1 = point_list[0:num-cut_point]
            temp2 = point_list[cut_point:num]
            list1.append(temp1)
            list2.append(temp2)

        #リストからvtpデータを作成
        for i in range(0,step,1):
            cut_point = round(num * cut_ratio * (i+1)/step)
            points1 = vtk.vtkPoints()
            points2 = vtk.vtkPoints()
            vertices1 = vtk.vtkCellArray()
            vertices2 = vtk.vtkCellArray()

            for j in range(0,num-cut_point,1):
                point1 = points1.InsertNextPoint(list1[i][j])
                #print(point1)
                vertices1.InsertNextCell(1)
                vertices1.InsertCellPoint(point1)

                point2 = points2.InsertNextPoint(list2[i][j])
                #print(point2)
                vertices2.InsertNextCell(1)
                vertices2.InsertCellPoint(point2)
                
                OutPoly1 = vtk.vtkPolyData()
                OutPoly1.SetPoints(points1)
                OutPoly1.SetVerts(vertices1)
                OutPoly2 = vtk.vtkPolyData()
                OutPoly2.SetPoints(points2)
                OutPoly2.SetVerts(vertices2)
            
            out_vtp1 = os.path.join(Outpath + "\\" + file_name + f"_1-{i+1}.vtp")
            out_vtp2 = os.path.join(Outpath + "\\" + file_name + f"_2-{i+1}.vtp")
            save_vtp(OutPoly1, out_vtp1)
            save_vtp(OutPoly2, out_vtp2)
            f.write(f"{file_name}_1-{i+1}\t0\t{num-cut_point}\n")
            f.write(f"{file_name}_2-{i+1}\t{cut_point}\t{num}\n")
            f.close
    
    # txtroot = os.path.join(Outpath + "\\devide_memo.txt")
    # f = open(txtroot, 'a')

    # rootlist = glob.glob(f'{Inpath}/*.vtp')
    # for Inroot in rootlist:
    #     #vtpデータ読み込み
    #     InPoly = vtp_data_loader(Inroot)
    #     file_name = Inroot.replace(Inpath, "")
    #     file_name = file_name.replace(".vtp", "")
    #     print(file_name)

    #     #vtpデータ読み込み
    #     InPoly = vtp_data_loader(Inroot)
    #     num = InPoly.GetNumberOfPoints()            #vtpのポイント数を取得
        
    #     f.write(f"{file_name}\t{num}\n")

    #     point_list = []
    #     for i in range(num):
    #         point = np.array(InPoly.GetPoint(i))
    #         point_list.append(point)
    #     print(len(point_list))

    #     #リスト生成
    #     cut_point = round(num * cut_ratio)
    #     print(cut_point)
    #     temp1 = point_list[0:num-cut_point]
    #     temp2 = point_list[cut_point:num]

    #     points1 = vtk.vtkPoints()
    #     points2 = vtk.vtkPoints()
    #     vertices1 = vtk.vtkCellArray()
    #     vertices2 = vtk.vtkCellArray()

    #     for i in range(0,num-cut_point,1):
    #         point1 = points1.InsertNextPoint(temp1[i])
    #         #print(point1)
    #         vertices1.InsertNextCell(1)
    #         vertices1.InsertCellPoint(point1)

    #         point2 = points2.InsertNextPoint(temp2[i])
    #         #print(point2)
    #         vertices2.InsertNextCell(1)
    #         vertices2.InsertCellPoint(point2)
            
    #         OutPoly1 = vtk.vtkPolyData()
    #         OutPoly1.SetPoints(points1)
    #         OutPoly1.SetVerts(vertices1)
    #         OutPoly2 = vtk.vtkPolyData()
    #         OutPoly2.SetPoints(points2)
    #         OutPoly2.SetVerts(vertices2)
        
    #     out_vtp1 = os.path.join(Outpath + "\\" + file_name + f"_1.vtp")
    #     out_vtp2 = os.path.join(Outpath + "\\" + file_name + f"_2.vtp")
    #     save_vtp(OutPoly1, out_vtp1)
    #     save_vtp(OutPoly2, out_vtp2)
    #     f.write(f"{file_name}_1\t0\t{num-cut_point}\n")
    #     f.write(f"{file_name}_2\t{cut_point}\t{num}\n")
    # f.close


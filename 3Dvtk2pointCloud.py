import numpy as np
import vtk
import glob
import os

def vtk2pointCloud(inputfilename, outputfilename,density):
    reader=vtk.vtkXMLImageDataReader()
    reader.SetFileName(inputfilename)
    reader.Update()
    image_data = reader.GetOutput()
    # 3次元のバイナリデータから点群を生成
    # 座標とスペーシングを取得
    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    bounds=image_data.GetBounds()
    extent=image_data.GetExtent()
    
    # ポイントクラウドのための空のリストを作成
    points = vtk.vtkPoints()
    vertices = vtk.vtkCellArray()

    # 3次元バイナリデータをスキャンしてポイントクラウドを生成
    dimensions = image_data.GetDimensions()
    if dimensions[2] == 1:
        for y in range(dimensions[1]):  # Y方向
            if y%density!=0:
                continue
            for x in range(dimensions[0]):  # X方向
                # 各座標のバイナリ値を取得
                if x%density!=0:
                    continue
                value = image_data.GetScalarComponentAsDouble(x, y, extent[4], 0)
                
                if value >= 1:
                    # 座標とスペーシングに基づいて点の位置を計算
                    point_x = origin[0] + x * spacing[0]
                    point_y = origin[1] + y * spacing[1]
                    point = points.InsertNextPoint(point_x, point_y,bounds[4] )
                    vertices.InsertNextCell(1)
                    vertices.InsertCellPoint(point)
                
             

    else :
        for z in range(dimensions[2]):  # Z方向
            if z%density!=0:
                continue
            for y in range(dimensions[1]):  # Y方向
                if y%density!=0:
                    continue
                for x in range(dimensions[0]):  # X方向
                    if y%density!=0:
                        continue
                    # 各座標のバイナリ値を取得
                    value = image_data.GetScalarComponentAsDouble(x, y, z, 0)
                    # バイナリ値が1ならポイントを追加
                    if value >= 1:
                        # 座標とスペーシングに基づいて点の位置を計算
                        point_x = origin[0] + x * spacing[0]
                        point_y = origin[1] + y * spacing[1]
                        point_z = origin[2] + z * spacing[2]
                        point = points.InsertNextPoint(point_x, point_y, point_z)
                        vertices.InsertNextCell(1)
                        vertices.InsertCellPoint(point)
                    
                
            

    # vtkPolyDataオブジェクトを作成してポイントをセット
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)

    # VTPファイルとして保存
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outputfilename)
    writer.SetInputData(polydata)
    writer.Write()

    print("VTPファイルが生成されました。")

def vti_reader(folder):
    vti_list = []
    name_list = []
    
    for file in glob.glob(folder + '*.vti'): 
        data = file
        vti_list.append(data)

    for file_name in os.listdir(folder):
        if file_name.endswith('.vti'):
            name = os.path.splitext(file_name)[0]
            name_list.append(name)
        
    return vti_list, name_list


#変換したいvtiデータが格納されたフォルダ
inpath="C:/Users/masuda2/source/repos/icpRegistration/202405Tanaka_conference/z159/"
#出力フォルダ
outpath = "C:/Users/masuda2/source/repos/icpRegistration/202405Tanaka_conference/z159/z159_density/"

#vtiデータ読み込み
vti_list, name_list = vti_reader(inpath)
print(vti_list)

#点群密度設定(例：density=3の時、点群1/3になる)
density=3
scale=1/density

for vti_data, name_data in zip(vti_list, name_list):
    vtk2pointCloud(vti_data, f"{outpath}{name_data}-den{density}.vtp", density)

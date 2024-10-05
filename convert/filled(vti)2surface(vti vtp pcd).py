import glob
import vtk
import numpy as np
from vtkmodules.util import numpy_support
import vtk
import os
import open3d as o3d
from utils_vtk import numpy_to_vtk, save_vtk, save_vtp, vtk_data_loader

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


def make_surface(vti_root):
    # VTIファイルの読み込み
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(vti_root)
    reader.Update()
    vti_data = reader.GetOutput()
    spa = vti_data.GetSpacing()

    # ラベルの輪郭を抽出
    contour = vtk.vtkContourFilter()
    contour.SetInputData(vti_data)
    # ラベル値を指定して輪郭を抽出（例: ラベル値が1の場合）
    contour.SetValue(0, 1)
    contour.Update()
    surface_vtp = contour.GetOutput()

    return surface_vtp, spa


def main(in_vti_path, out_path):    
    outvti_path = os.path.join(out_path + "\\VTI(surface)")
    outvtp_path = os.path.join(out_path + "\\VTP(surface)")
    outpcd_path = os.path.join(out_path + "\\PCD(surface)")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    if not os.path.exists(outvti_path):
        os.mkdir(outvti_path)
    if not os.path.exists(outvtp_path):
        os.mkdir(outvtp_path)
    if not os.path.exists(outpcd_path):
        os.mkdir(outpcd_path)

    vti_rootlist = glob.glob(f"{in_vti_path}/*Y-deform-rev(filled)_padding(20)_closing(10)_15-0_10-2.vti")    

    for vti_root in vti_rootlist:
        print(vti_root)
        surface_vtp, spa = make_surface(vti_root)

        base_name = os.path.basename(vti_root)        
        file_name = os.path.splitext(base_name)[0]
        file_name = file_name.replace("(filled)", "")
        print(file_name)        

        vtp_outroot = os.path.join(outvtp_path + "\\" + file_name + "(surface).vtp")
        vti_outroot = os.path.join(outvti_path + "\\" + file_name + "(surface).vti")
        pcd_outroot = os.path.join(outpcd_path + "\\" + file_name + "(surface).pcd")
        #print(vtp_outroot)

        #出力
        save_vtp(surface_vtp, vtp_outroot)
        pointList, out_vti = vtp_to_list_vti(surface_vtp, root=None, spacing=spa)
        save_vtk(out_vti, vti_outroot)
        out_pcd = o3d.geometry.PointCloud()
        out_pcd.points = o3d.utility.Vector3dVector(pointList)    
        o3d.io.write_point_cloud(pcd_outroot, out_pcd)

    

if __name__ == '__main__':  
    in_vti_path = r"D:\takahashi_k\registration(model)\forUSE\original\filled"
    out_path = r"D:\takahashi_k\registration(model)\forUSE\original"    
   
    main(in_vti_path, out_path)
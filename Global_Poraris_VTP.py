#registration roughly by polaris data
#2値データ
import numpy as np
import vtk
from utils.utils_vtk import save_vtp, vtk_data_loader, vti_to_vtp, vtp_data_loader
import os
import glob

def regi_by_mtx(mtx, InPoly):
    p_P = np.ones((4, 1), dtype=float)          #行列(1で初期化)
    Gpoints = vtk.vtkPoints()                   #グローバル座標系
    Gvertices = vtk.vtkCellArray()
    g_P = np.empty((4, 1), dtype=float)         #行列(空配列)

    #位置合わせまえの点群の座標データ取得
    num = InPoly.GetNumberOfPoints()
    print(num)
    for i in range(num):
        coordinate = InPoly.GetPoint(i)
        point_x = coordinate[0]
        point_y = coordinate[1]
        point_z = coordinate[2]

        #グローバル座標系計算
        p_P[0,0] = point_x
        p_P[1,0] = point_y
        p_P[2,0] = point_z
        g_P = np.dot(mtx, p_P)              #行列の積
        #グローバル座標系座標系
        Gpoint = Gpoints.InsertNextPoint(g_P[0,0], g_P[1,0], g_P[2,0])
        Gvertices.InsertNextCell(1)
        Gvertices.InsertCellPoint(Gpoint)

    OutPoly = vtk.vtkPolyData()
    OutPoly.SetPoints(Gpoints)
    OutPoly.SetVerts(Gvertices)

    return OutPoly


# if __name__ == '__main__':
#     """pathの設定  mtxファイルの拡張子を.txtに変更すること"""
#     path = r"D:\takahashi_k\temporary space"
#     vtiroot = os.path.join(path + "\\Ogawa_IM_0134-seg.vti")                    #出力のプローブ座標系のvtiデータ
#     Probe_vtproot = os.path.join(path + "\\Ogawa_IM_0134-seg_rev.vtp")              #出力のプローブ座標系のvtpデータ
#     Global_vtproot = os.path.join(path + "\\Ogawa_IM_0134-seg_Global_rev.vtp")      #出力のグローバル座標系のvtpデータ
#     mtxroot = os.path.join(path + "\\IM_0134_Phantom2Volume.txt")

#     #vtiデータ読み込み
#     vti_array, spa, ori = vtk_data_loader(vtiroot)
#     #vti2vtp
#     probe_poly = vti_to_vtp(vti_array, spa, ori)
#     #save polydata
#     save_vtp(probe_poly, Probe_vtproot)

#     #同時変換行列を読み込み
#     f = open(mtxroot, 'r')
#     mtx = f.read()
#     mtx = mtx.replace(",", "")
#     mtx = mtx.split()
#     mtx = np.array(mtx, dtype=float)
#     mtx = mtx.reshape(4,4)
#     print(mtx)
#     print(type(mtx))
#     f.close()

#     Global_poly = regi_by_mtx(mtx, probe_poly)
#     # VTPファイルとして保存
#     save_vtp(Global_poly, Global_vtproot)

if __name__ == '__main__':
    """pathの設定  mtxファイルの拡張子を.txtに変更すること"""
    path = r"D:\takahashi_k\database\us\20240502_YAMAGUCHI"
    Probe_vtppath = os.path.join(path + "\\Annotation\\VTP(surface)")              #入力のプローブ座標系のvtpデータ
    Global_vtppath = os.path.join(path + "\\Global(0)\\VTP(surface)")      #出力のグローバル座標系のvtpデータ
    mtxpath = os.path.join(path + "\\Matrix")

    rootlist = glob.glob(f'{Probe_vtppath}/*.vtp')
    num = 1
    for vtp_root in rootlist:
        print(num)
        probe_poly = vtp_data_loader(vtp_root)
        filename = vtp_root.replace(Probe_vtppath, "")
        filename = filename.replace(".vtp", "")
        mtxroot = os.path.join(mtxpath + f"\\Volume{num}.csv")
        Global_vtproot = os.path.join(Global_vtppath + filename + "_Global.vtp")

        #同時変換行列を読み込み
        # mtx = np.loadtxt(mtxroot)
        # print(mtx)
        f = open(mtxroot, 'r')
        mtx = f.read()
        mtx = mtx.replace(",", "")
        mtx = mtx.split()
        mtx = np.array(mtx, dtype=float)
        mtx = mtx.reshape(4,4)
        print(mtx)
        print(type(mtx))
        f.close()

        #probe_poly = vtp_data_loader(vtp_root)
        Global_poly = regi_by_mtx(mtx, probe_poly)
        # VTPファイルとして保存
        save_vtp(Global_poly, Global_vtproot)
        num += 1

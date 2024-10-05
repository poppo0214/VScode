import os
import cv2
import sys
import datetime
from utils.utils_vtk import vtk_data_loader
from skimage.measure import profile_line
import matplotlib.pyplot as plt
import subprocess
inpath = "d:\\takahashi_k\\database\\us\\senior\\origin"
outpath = "d:\\takahashi_k\\lineprofile"
infile = "Watanabe_0024_Origin"
inroot = os.path.join(inpath + "\\" + infile + ".vti")
#outslice = os.path.join(outpath + "\\" + infile + "_slice.png")
outtxt = os.path.join(outpath + "\\" + infile + "position.txt")
#outprofile = os.path.join(outpath + "\\" + infile + "_lineprofile.png")
paraview = r"C:\Program Files\ParaView 5.8.1-Windows-Python3.7-msvc2015-64bit\bin\paraview.exe"


def posision_txt(x, y):
    #座標データをテキストファイルに出力
    f = open(outtxt, 'a')
    f.write(f"{x}\t{y}\t")
    f.close()

def mouse_event(event, x, y, flags, params):
    #左クリック時（始点）
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        posision_txt(x, y)
    #右クリック時（終点）
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(x, y)
        posision_txt(x, y)
    

if __name__ == '__main__':
    #paraviewを起動し、スライスを選択
    subprocess.Popen([paraview, inroot])
    print("choose slise")
    axis = input("input axis>>>")
    num = int(input("input slzice number>>>"))
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    d = now.strftime('%Y%m%d%H%M%S')
    outslice = os.path.join(outpath + "\\" + infile + f"_slice({axis}{num})_{d}.png")
    outprofile = os.path.join(outpath + "\\" + infile + f"_lineprofile({axis}{num})_{d}.png")

    array, spa, ori = vtk_data_loader(inroot) 
    if axis == "x":
        slice = array[num, :, :]
    elif axis == "y":
        slice = array[:, num, :]
    elif axis == "z":
        slice = array[:, :, num]
    else:
        print("error")
        sys.exit()
    
    slice = slice.T                 #opencvだと軸が入れ替わってしまうので、転置行列にしておく
    cv2.imwrite(outslice, slice)
    slice = cv2.imread(outslice, cv2.IMREAD_GRAYSCALE)
    #スライスを画面上に表示し、プロファイルラインの始点と終点を選択
    print("Left click on the starting point, right click on the ending point\nPress some key when you are done.")
    cv2.imshow('slice', slice)  #スライスをGUI表示
    cv2.setMouseCallback('slice', mouse_event)
    cv2.waitKey(0)              #何らかのkeyboard入力があるまで待機

    #プロファイルラインを作成
    is_file = os.path.isfile(outtxt)                                #座標データが入ったテキストファイルを読み込み
    if is_file:                                                     #座標データ取得
        f = open(outtxt, 'r')
        list = f.read()
        f.close()
        list = list.split()
        # if len(list) != 4:
        #     print("ERROR: point is more than 2")
        #     sys.exit()
        x_start = int(list[0])
        y_start = int(list[1])
        x_end = int(list[2])
        y_end = int(list[3])
        #profile = profile_line(slice, (x_start, y_start), (x_end, y_end), linewidth=1) 
        profile = profile_line(slice, (y_start, x_start), (y_end, x_end), linewidth=1)      #プロファイルライン画像を生成
        fig, ax = plt.subplots(ncols=2,figsize=(8,4))
        ax[0].imshow(slice, cmap="gray")
        ax[0].plot([x_start,x_end],[y_start,y_end],'r-',lw=1)
        # ax[0].annotate("start", xy=(x_start, y_start), size=10, xytext=(x_start-10, y_start), color="red", arrowprops=dict())
        # ax[0].annotate("end", xy=(x_end, y_end), size=10, xytext=(x_end+2, y_end), color="red", arrowprops=dict())
        ax[0].annotate("start", xy=(x_start, y_start), size=10, xytext=(x_start-40, y_start), color="red",)
        ax[0].annotate("end", xy=(x_end, y_end), size=10, xytext=(x_end+2, y_end), color="red")
        ax[1].plot(profile)
        ax[1].set_ylim(0, 255)
        ax[1].set_title('data points = '+str(profile.shape[0])+'')
        ax[1].set_xlabel("<-start  end->")
        ax[1].set_ylabel("brightness value")
        plt.tight_layout()
        plt.savefig(outprofile,dpi=100)

        os.remove(outtxt)
    






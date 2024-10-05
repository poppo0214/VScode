import numpy as np
import os
import cv2
import sys
import datetime
import math
from utils.utils_vtk import vtk_data_loader
from skimage.measure import profile_line
import matplotlib.pyplot as plt
from tkinter import messagebox
import subprocess
import csv
inpath = "d:\\takahashi_k\\database\\us\\peer\\origin"
outpath = "d:\\takahashi_k\\model"
infile = "Shinoda_0131-Origin"
inroot = os.path.join(inpath + "\\" + infile + ".vti")
#ozzutslice = os.path.join(outpath + "\\" + infile + "_slice.png")
profiletxt = os.path.join(outpath + "\\" + infile + "_position.txt")
modeltxt = os.path.join(outpath + "\\" + infile + "_xposition.txt")
walltsv = os.path.join(outpath + "\\wallmodel.csv")
nonwalltsv = os.path.join(outpath + "\\nonwallmodel.csv")
#outprofile = os.path.join(outpath + "\\" + infile + "_lineprofile.png")
paraview = r"C:\Program Files\ParaView 5.8.1-Windows-Python3.7-msvc2015-64bit\bin\paraview.exe"
rot = True

def position_txt(x, y):
    #座標データをテキストファイルに出力
    f = open(profiletxt, 'a')
    f.write(f"{x}\t{y}\t")
    f.close()

def x_txt(x):
    f = open(modeltxt, 'a')
    f.write(f"{x}\t")
    f.close()

def mouse_event(event, x, y, flags, params):
    #左クリック時（始点）
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        position_txt(x, y)
    #右クリック時（終点）
    elif event == cv2.EVENT_RBUTTONDOWN:
        print(x, y)
        position_txt(x, y)

def on_key(event):
    global ind
    if event.key == 'right':
        #move = 1
        ind += 1
        ind %= n
    elif event.key == 'left':
        #move = -1
        ind += -1
        ind %= n
    elif event.key == '1':
        print('wall')
        x_txt(1)
    elif event.key == '2':
        print('nonwall')
        x_txt(2)
    elif event.key == 'enter':
        print(x[ind], y[ind])
        x_txt(x[ind])
    else:
        return

    # インデックスの更新
    # ind += move
    # ind %= n  # グラフの端に行くと戻るようにする

    # カーソルとタイトルの更新
    cur_v.set_xdata([x[ind]])
    cur_h.set_ydata([y[ind]])
    cur_point.set_data([x[ind]], [y[ind]])
    ax.set_title('index: {}, x = {}, y = {}'.format(
                                    ind, round(x[ind], 4), round(y[ind], 4)))
    plt.draw()

def analyze(profile, Ph, Ed, outprofile):
    f = open(modeltxt, 'r')
    list = f.read()
    f.close()
    list = list.split()
    list = [int(s) for s in list]
    
    if len(list) != 5:
        print("WARNING: The fifth data from the back is used to determine the presence or absence of vessel walls, and up to the fourth data from the back is used for luminance analysis.")
        ans = input("Are you sure?\ty/n>>>")
        if ans == "y":
            list = list[len(list)-5 : len(list)]
        else:
            os.remove(modeltxt)
            os.remove(outprofile)
            sys.exit()

    flag = list[0]
    list = np.delete(list, 0)  

    T1 = int(list[0])
    B1 = int(list[1])
    B2 = int(list[2])
    T2 = int(list[3])
    m1 = int((T1+B1)/2)
    m2 = int((T2+B2)/2)
    r = int((m2-m1)/2)
    print(r)
    
    
    center = int(((T2+B2)/2 + (T1+B1)/2)/2)
    r1 = Ed * (len(profile[m1:center])/Ph)         #半径[mm]を計算  
    r2 = Ed * (len(profile[center:m2])/Ph)   


    #血管壁なし
    if flag == 2:
        E1 = m1 - r
        E2 = m2 + r

        if E1 < 0:
            print("ERROR: Select a range sufficiently larger than the vessel diameter")
            os.remove(modeltxt)      
            os.remove(outprofile)
            sys.exit()
        elif E2 > len(profile):
            print("ERROR: Select a range sufficiently larger than the vessel diameter")
            os.remove(modeltxt)
            os.remove(outprofile)
            sys.exit()

        model1 = np.flipud(profile[E1:center])
        model2 = profile[center:E2]
        model1 = np.insert(model1, 0, r1)               #モデルの配列の頭に長さを追加
        model2 = np.insert(model2, 0, r2)
    # print(model1)
    # print(model2)       
        # with open(nonwalltsv, "a") as f:
        #     np.savetxt(f, model1, delimiter=',')
        #     np.savetxt(f, model2, delimiter=',')
        f = open(nonwalltsv, 'a', newline='')
        writer = (f)
        writer.writerow(model1)
        writer.writerow(model2)
        f.close

    #血管壁あり
    elif flag == 1:
        E1 = T1 - r
        E2 = T2 + r     

        if E1 < 0:
            print("ERROR: Select a range sufficiently larger than the vessel diameter")
            os.remove(modeltxt)      
            os.remove(outprofile)
            sys.exit()
        elif E2 > len(profile):
            print("ERROR: Select a range sufficiently larger than the vessel diameter")
            os.remove(modeltxt)
            os.remove(outprofile)
            sys.exit()

        model1 = np.flipud(profile[E1:center])
        model2 = profile[center:E2]
        model1 = np.insert(model1, 0, r1)               #モデルの配列の頭に長さを追加
        model2 = np.insert(model2, 0, r2)     
        # with open(walltsv, "a") as f:
        #     np.savetxt(f, model1, delimiter=',')
        #     np.savetxt(f, model2, delimiter=',')
        f = open(walltsv, 'a', newline='')
        writer = csv.writer(f)
        writer.writerow(model1)
        writer.writerow(model2)
        f.close
    else:
        print("ERROR: Enter 1 for vessels with vessel walls and 2 for vessels without vessel walls first")
        os.remove(modeltxt)
        os.remove(outprofile)
        sys.exit()


if __name__ == '__main__':
    #paraviewを起動し、スライスを選択
    #subprocess.Popen([paraview, inroot])
    print("choose slise")
    axis = input("input axis>>>")
    num = int(input("input slice number>>>"))
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    d = now.strftime('%Y%m%d%H%M%S')
    outslice = os.path.join(outpath + "\\" + infile + f"_slice({axis}{num})_{d}.png")
    outprofile = os.path.join(outpath + "\\" + infile + f"_lineprofile({axis}{num})_{d}.png")

    array, spa, ori = vtk_data_loader(inroot) 
    if axis == "x":
        slice = array[num, :, :]
        Sh = spa[1]
        Sv = spa[2]
    elif axis == "y":
        slice = array[:, num, :]
        Sh = spa[0]
        Sv = spa[2]
    elif axis == "z":
        slice = array[:, :, num]
        Sh = spa[0]
        Sv = spa[1]
    else:
        print("error")
        sys.exit()
    
    
    cv2.imwrite(outslice, slice)
    slice = cv2.imread(outslice, cv2.IMREAD_GRAYSCALE)
    if rot:
        slice = slice.T                 #軸の入れ替え
    os.remove(outslice)
    #スライスを画面上に表示し、プロファイルラインの始点と終点を選択
    print("Left click on the starting point, right click on the ending point\nPress some key when you are done.")
    cv2.namedWindow("slice", cv2.WINDOW_NORMAL)
    cv2.imshow('slice', slice)  #スライスをGUI表示
    cv2.setMouseCallback('slice', mouse_event)
    cv2.waitKey(0)              #何らかのkeyboard入力があるまで待機

    #プロファイルラインを作成
    is_file = os.path.isfile(profiletxt)                                #座標データが入ったテキストファイルを読み込み
    if is_file:                                                     #座標データ取得
        f = open(profiletxt, 'r')
        list = f.read()
        f.close()
        list = list.split()
        list = [int(s) for s in list]

        if len(list) != 4:
            print("ERROR: point is more than 2")
            #os.remove(outslice)
            os.remove(profiletxt)
            sys.exit()
        
        x_start = int(list[0])
        y_start = int(list[1])
        x_end = int(list[2])
        y_end = int(list[3])
        Ph = abs(x_end - x_start)
        Pv = abs(y_end - y_start)
        if Ph == 0:
            Ph = 1
        elif Pv == 0:
            Pv = 1
        Ed = math.sqrt(pow(Ph*Sh, 2) + pow(Pv*Sv, 2))
        profile = profile_line(slice, (y_start, x_start), (y_end, x_end), linewidth=1)      #プロファイルライン画像を生成
        fig, ax = plt.subplots(ncols=2,figsize=(8,4))
        ax[0].imshow(slice, cmap="gray")
        ax[0].plot([x_start,x_end],[y_start,y_end],'r-',lw=1)
        ax[0].annotate("start", xy=(x_start, y_start), size=10, xytext=(x_start-40, y_start), color="red",)
        ax[0].annotate("end", xy=(x_end, y_end), size=10, xytext=(x_end+2, y_end), color="red")
        ax[1].plot(profile)
        ax[1].set_ylim(0, 255)
        ax[1].set_title('data points = '+str(profile.shape[0])+'')
        ax[1].set_xlabel("<-start  end->")
        ax[1].set_ylabel("brightness value")
        plt.tight_layout()
        plt.savefig(outprofile,dpi=100)
        os.remove(profiletxt)        
        plt.clf()


        fig, ax = plt.subplots()
        y = profile
        x = np.arange(len(y))
        n = len(x)
        ind = 0    # カーソル位置のインデックス
        cur_v = ax.axvline(x[ind], color='k', linestyle='--', linewidth=0.5)
        cur_h = ax.axhline(y[ind], color='k', linestyle='--', linewidth=0.5)
        ax.plot(x, y, "o-",picker=15)
        cur_point, = ax.plot(x[ind], y[ind], color='k', markersize=10, marker='o')
        ax.set_title('a vessel with vascular walls -> 1\na vessel without vascular walls -> 2\nindex: {}, x = {}, y = {}'.format(
                                        ind, round(x[ind], 4), round(y[ind], 4)))
        
        messagebox.showinfo('確認', 'a vessel with vascular walls -> 1\na vessel without vascular walls -> 2')
        #plt.show()
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        #os.remove(outslice)
    



    #モデル
    is_file = os.path.isfile(modeltxt)  
    if is_file:                                                     #座標データ取得
        analyze(profile, Ph, Ed, outprofile)
        os.remove(modeltxt)


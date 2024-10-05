import numpy as np
import os
import csv
import statistics

def devide(inpath, infile):    
    inroot = os.path.join(inpath + "\\" + infile + ".csv")
    all = []
    with open(inroot) as f:
        reader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            all.append(row)
    f.close()

    for i in range(len(all)):
        list = all[i]
        r = list[0]
        if r>=0.5 and r<1.5:         #0.5-1.5(1)
            f1 = open(os.path.join(inpath + "\\" + infile + "_1(0.5-1.5).csv"), 'a', newline='')
            writer1 = csv.writer(f1)
            writer1.writerow(list)
            f1.close
        elif r>=1.5 and r<2.5:       #1.5-2.5(1)
            f2 = open(os.path.join(inpath + "\\" + infile + "_2(1.5-2.5).csv"), 'a', newline='')
            writer2 = csv.writer(f2)
            writer2.writerow(list)
            f2.close
        elif r>=2.5 and r<3.5:      #2.5-3.5(1)
            f3 = open(os.path.join(inpath + "\\" + infile + "_3(2.5-3.5).csv"), 'a', newline='')
            writer3 = csv.writer(f3)
            writer3.writerow(list)
            f3.close
        elif r>=3.5 and r<4.5:
            f4 = open(os.path.join(inpath + "\\" + infile + "_4(3.5-4.5).csv"), 'a', newline='')
            writer4 = csv.writer(f4)
            writer4.writerow(list)
            f4.close
        elif r>=4.5 and r<6:
            f5 = open(os.path.join(inpath + "\\" + infile + "_5(4.5-6).csv"), 'a', newline='')
            writer5 = csv.writer(f5)
            writer5.writerow(list)
            f5.close
        elif r>=6 and r<8:
            f7 = open(os.path.join(inpath + "\\" + infile + "_7(6-8).csv"), 'a', newline='')
            writer7 = csv.writer(f7)
            writer7.writerow(list)
            f7.close
        elif r>=8 and r<10:
            f9 = open(os.path.join(inpath + "\\" + infile + "_9(8-10).csv"), 'a', newline='')
            writer9 = csv.writer(f9)
            writer9.writerow(list)
            f9.close
        elif r>=10 and r<12.5:
            f11 = open(os.path.join(inpath + "\\" + infile + "_11(10-12.5).csv"), 'a', newline='')
            writer11 = csv.writer(f11)
            writer11.writerow(list)
            f11.close
        elif r>=12.5 and r<15.5:
            f14 = open(os.path.join(inpath + "\\" + infile + "_14(12.5-15.5).csv"), 'a', newline='')
            writer14 = csv.writer(f14)
            writer14.writerow(list)
            f14.close
        elif r>=15.5 and r<18.5:
            f17 = open(os.path.join(inpath + "\\" + infile + "_17(15.5-18.5).csv"), 'a', newline='')
            writer17 = csv.writer(f17)
            writer17.writerow(list)
            f17.close
        elif r>=18.5 and r<22.5:
            f20 = open(os.path.join(inpath + "\\" + infile + "_20(18.5-22.5).csv"), 'a', newline='')
            writer20 = csv.writer(f20)
            writer20.writerow(list)
            f20.close
        elif r>=22.5 and r<27.5:
            f25 = open(os.path.join(inpath + "\\" + infile + "_25(22.5-27.5).csv"), 'a', newline='')
            writer25 = csv.writer(f25)
            writer25.writerow(list)
            f25.close
        elif r>=27.5 and r<32.5:
            f30 = open(os.path.join(inpath + "\\" + infile + "_30(27.5-32.5).csv"), 'a', newline='')
            writer30 = csv.writer(f30)
            writer30.writerow(list)
            f30.close

def process(inpath, infile, r):
    inroot = os.path.join(inpath + "\\" + infile + r + ".csv")
    outroot = os.path.join(inpath + "\\" + infile + r + "_points.csv")
    all = []
    maxV_list = []
    minV_list = []
    edgeV_list = []

    with open(inroot) as fin:
        reader = csv.reader(fin,quoting=csv.QUOTE_NONNUMERIC)
        for row in reader:
            all.append(row)
    fin.close()

    max_num = 0
    for i in range(len(all)):
        list1 = all[i]
        del list1[0]
        
        point_num = len(list1)                   #モデルの座標数                             #半径の値消去
        if point_num > max_num:
            max_num = point_num

        edgeV_list.append(list1[point_num-1])    #端の点（肝実質）の輝度取得
        maxv = max(list1)                        #最大輝度（血管壁）取得
        maxV_list.append(maxv)
        minv = min(list1)                        #最小輝度（血管内腔）取得
        minV_list.append(minv)
        
    print(max_num)
    #オフセット値を計算
    offset = (statistics.mean(edgeV_list)-statistics.mean(minV_list)) / (statistics.mean(maxV_list)-statistics.mean(minV_list))
    print(offset)

    for j in range(len(all)):
        list2 = all[j]
        print(j, list2)
        #del list2[0]
        print(list2)
        maxv = max(list2)                       
        minv = min(list2)
        print(j, maxv-minv)                        
        #正規化
        norm_list = [n - minv for n in list2]
        norm_list = [n/(maxv-minv) for n in norm_list]
        #オフセット        
        offset_list = [n - offset for n in norm_list]

        for k in range(len(list2)):
            step = max_num/len(list2)
            point = [step*k, list2[k], norm_list[k], offset_list[k]]

            fout = open(outroot, "a", newline='')
            writer = csv.writer(fout)
            writer.writerow(point)
            if k != 0:
                point_neg =[-step*k, list2[k], norm_list[k], offset_list[k]]
                writer.writerow(point_neg)
            fout.close
    


if __name__ == '__main__':
    inpath = "d:\\takahashi_k\\model\\analyze\\wall"
    outpath = inpath
    infile = "wallmodel"
    r_list = ["_1(0.5-1.5)", "_2(1.5-2.5)", "_3(2.5-3.5)", "_4(3.5-4.5)", "_5(4.5-6)", "_7(6-8)", "_9(8-10)", "_11(10-12.5)", "_14(12.5-15.5)", "_17(15.5-18.5)", "_20(18.5-22.5)", "_25(22.5-27.5)", "_30(27.5-32.5)"]

    #devide(inpath, infile)
    for r in r_list:
        process(inpath, infile, r)
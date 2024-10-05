import cv2
import numpy as np
txtroot = "D:/takahashi_k/2Dtest/bigger/width(10).txt"
th = 10
f = open(txtroot, "w")
for j in range(1, 31, 1):
    array = cv2.imread(f"D:/takahashi_k/2Dtest/bigger/frg({j}).png", cv2.IMREAD_GRAYSCALE)    
    temp1 = 0
    width = []
    for i in range(0, 100, 1):
        temp2 = array[1][i]
        if temp1 <= th and th < temp2:
            flag = i
        elif temp1 > th and th >= temp2:
            width.append(i-flag) 
        temp1 = temp2
    maxwidth = max(width)
    f.write(f"{j}\t{maxwidth}\n")
f.close

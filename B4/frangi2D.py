from skimage.filters import frangi, hessian
import cv2
import numpy as np

sigmas = range(1, 20, 2)
array = cv2.imread("D:\\takahashi_k\\new_function\\2d\\BOAT.bmp", cv2.IMREAD_GRAYSCALE)

#frangifilter
output = np.zeros_like(array)
for sigma in sigmas:
    temp = frangi(array, sigmas=range(sigma, sigma+1, 1), black_ridges=True)
    output = np.maximum(output, temp)
output = output*255
print(np.max(output))

#出力
cv2.imwrite(f"D:\\takahashi_k\\new_function\\2d\\BOAT_frg.bmp", output)
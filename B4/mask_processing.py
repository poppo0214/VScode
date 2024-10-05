from warnings import warn

import numpy as np
from scipy import linalg

from skimage._shared.utils import _supported_float_type, check_nD, deprecated
from skimage.feature.corner import hessian_matrix, hessian_matrix_eigvals
from skimage.util import img_as_float
from skimage.filters import frangi
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import os
import numpy as np
from utils.labeling import labeling
import cv2

maxS = 5
In_path = 'd:\\takahashi_k\\simulation\\add_noise\\vessel\\'
#gaussis = range(1, 11, 1)
Out_path = In_path
origin = "gauss(1)inv(150-0)_speckle(1)rev(1_4)_gauss(10)_rev"      #正解データ
close = "noise(rev)_frg(1,2,1)_close(5)"                            #マスク演算するデータ
Out_name = f"noise_close(5)_mask({maxS}, vp)"


def frangi(image, sigmas=range(1, 10, 2), scale_range=None,
           scale_step=None, alpha=0.5, beta=0.5, gamma=None,
           black_ridges=True, mode='reflect', cval=0):

    print("sigmas = ", list(sigmas))
    if scale_range is not None and scale_step is not None:
        warn('Use keyword parameter `sigmas` instead of `scale_range` and '
             '`scale_range` which will be removed in version 0.17.',
             stacklevel=2)
        sigmas = np.arange(scale_range[0], scale_range[1], scale_step)

    check_nD(image, [2, 3])  # Check image dimensions.
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:  # Normalize to black ridges.
        image = -image

    # Generate empty array for storing maximum value
    # from different (sigma) scales
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:  # Filter for all sigmas.
        print(f"sigma = {sigma}")
        eigvals = hessian_matrix_eigvals(hessian_matrix(
            image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True))
        # Sort eigenvalues by magnitude.
        eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
        lambda1 = eigvals[0]
        if image.ndim == 2:
            lambda2, = np.maximum(eigvals[1:], 1e-10)
            r_a = np.inf  # implied by eq. (15).
            r_b = abs(lambda1) / lambda2  # eq. (15).
        else:  # ndim == 3
            lambda2, lambda3 = np.maximum(eigvals[1:], 1e-10)
            r_a = lambda2 / lambda3  # eq. (11).
            r_b = abs(lambda1) / np.sqrt(lambda2 * lambda3)  # eq. (10).
        s = np.sqrt((eigvals ** 2).sum(0))  # eq. (12).
        if gamma is None:
            gamma = s.max() / 2
            if gamma == 0:
                gamma = 1  # If s == 0 everywhere, gamma doesn't matter.
        # Filtered image, eq. (13) and (15).  Our implementation relies on the
        # blobness exponential factor underflowing to zero whenever the second
        # or third eigenvalues are negative (we clip them to 1e-10, to make r_b
        # very large).
        vals = 1.0 - np.exp(-r_a**2 / (2 * alpha**2))  # plate sensitivity
        vals *= np.exp(-r_b**2 / (2 * beta**2))  # blobness
        vals *= 1.0 - np.exp(-s**2 / (2 * gamma**2))  # structuredness
        filtered_max = np.maximum(filtered_max, vals)
        #histogram(vals, sigma)
    return filtered_max  # Return pixel-wise max over all sigmas.

def volume_processing(array, th = None):
    print(f"volume processing({th})")
    labeledarray, N = labeling(array)

    areas = []
    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        areas.append(count)

    areas = np.array(areas)
    areas = np.uint16(areas)

    if th == None:                      #大津の2値化による閾値の設定
        th, __ =  cv2.threshold(areas, 0, 1, cv2.THRESH_OTSU)
        print(f"th by OTSU = {th}")                            

    output = np.zeros_like(array)

    for num in range(1,N+1):
        count = np.count_nonzero(labeledarray == num)
        if count >= th:
            temp = np.where(labeledarray == num , 1, 0)
            output = output + temp

    output = output * array
    return(output, th)

if __name__ == '__main__':
    origin_root = os.path.join(In_path + origin + '.vti')
    close_root = os.path.join(In_path + close + '.vti')
    Out_root = os.path.join(Out_path + Out_name + '.vti')

    print("loading vti file...")
    origin_array, spa, ori = vtk_data_loader(origin_root)
    close_array, spa, ori = vtk_data_loader(close_root)

    print("frangi")
    max_array = frangi(origin_array, sigmas=range(maxS, maxS+1, 1), black_ridges=True)
    max_array, __ = volume_processing(max_array, None)
    mask = max_array > 0
    output = mask * close_array

    #output = frangi(array, sigmas=sgms, black_ridges=False)
    output = output * (255/np.max(output))

    #save as vti
    print("saving as vti...")
    output = numpy_to_vtk(output, spa, ori)
    save_vtk(output, Out_root)
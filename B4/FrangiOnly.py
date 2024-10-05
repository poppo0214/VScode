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
import matplotlib.pyplot as plt
import glob



def histogram(array, sigma):
    # Graph Settings            
    plt.title(f"distribution of sigma({sigma})", fontsize=20)  
    plt.xlabel("brightness value", fontsize=12)   
    plt.ylabel("the number of voxel", fontsize=12) 

    rearray = array.flatten()                             #3次元配列を1次元に平滑化
    n, bins, _ = plt.hist(rearray, bins=np.linspace(0, 1, 20), log=True)               #n : 配列または配列のリスト,ヒストグラムのビンの値    bins：いくつの階級に分割するか
    
    xs = (bins[:-1] + bins[1:])/2
    ys = n.astype(int)

    for x, y in zip(xs, ys):
        plt.text(x, y, str(y), horizontalalignment="center", size=6, rotation=90)
    
    histroot = os.path.join(In_path + f'histogram\\step=1\\array{sigma}.png')
    plt.savefig(histroot)
    plt.clf()


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
        # Sort eigenvalues by magnitude
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

if __name__ == '__main__':
    In_path = r'd:\takahashi_k\database\us\origin'
    Out_path = r'd:\takahashi_k\database\us\frangi'
    sgms = range(1,17,2)
            
    rootlist = glob.glob(f'{In_path}/*.vti')
    for root in rootlist:
        print("loading vti file...")
        print("root")
        array, spa, ori = vtk_data_loader(root)

        print("frangi")
        output = np.zeros_like(array)
        for sigma in sgms:
            temp = frangi(array, sigmas=range(sigma, sigma+1, 1), black_ridges=True)
            output = np.maximum(output, temp)
            temp = temp * (255/np.max(temp))

            filename = root.replace(In_path, "")
            filename = filename.replace(".vti", "")
            Out_root = os.path.join(Out_path + f"\\{sigma}" + filename + f"_frg({sigma}).vti")
            print(Out_root)
            temp_vtk = numpy_to_vtk(temp, spa, ori)
            save_vtk(temp_vtk, Out_root)

        #output = frangi(array, sigmas=sgms, black_ridges=False)
        output = output * (255/np.max(output))

        #save as vti
        print("saving as vti...")
        filename = root.replace(In_path, "")
        filename = filename.replace(".vti", "")
        Out_root = os.path.join(Out_path + f"\\(1,17,2)" + filename + "_frg(1,17,2).vti")
        print(Out_root)
        output_vtk = numpy_to_vtk(output, spa, ori)
        save_vtk(output_vtk, Out_root)
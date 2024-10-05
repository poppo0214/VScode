import math
import functools
from itertools import combinations_with_replacement
from warnings import warn

import numpy as np
from numpy.core.multiarray import normalize_axis_index
from collections.abc import Iterable
import numbers
from scipy import ndimage as ndi

from skimage._shared.utils import _supported_float_type, check_nD, deprecated
from skimage.feature.util import img_as_float
from skimage._shared.filters import gaussian
from scipy.ndimage import _ni_docstrings
from scipy.ndimage import _ni_support
from scipy.ndimage import _nd_image

import os
import cv2
import sympy as sp
from utils.utils_vtk import vtk_data_loader, numpy_to_vtk, save_vtk
import matplotlib.pyplot as plt
import glob

def _invalid_origin(origin, lenw):
    return (origin < -(lenw // 2)) or (origin > (lenw - 1) // 2)

def _complex_via_real_components(func, input, weights, output, cval, **kwargs):
    """実数畳み込みの線形結合による複素畳み込み。"""
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input and complex_weights:
        # real component of the output
        func(input.real, weights.real, output=output.real,
             cval=np.real(cval), **kwargs)
        output.real -= func(input.imag, weights.imag, output=None,
                            cval=np.imag(cval), **kwargs)
        # imaginary component of the output
        func(input.real, weights.imag, output=output.imag,
             cval=np.real(cval), **kwargs)
        output.imag += func(input.imag, weights.real, output=None,
                            cval=np.imag(cval), **kwargs)
    elif complex_input:
        func(input.real, weights, output=output.real, cval=np.real(cval),
             **kwargs)
        func(input.imag, weights, output=output.imag, cval=np.imag(cval),
             **kwargs)
    else:
        if np.iscomplexobj(cval):
            raise ValueError("Cannot provide a complex-valued cval when the "
                             "input is real.")
        func(input, weights.real, output=output.real, cval=cval, **kwargs)
        func(input, weights.imag, output=output.imag, cval=cval, **kwargs)
    return output

@_ni_docstrings.docfiller
def correlate1d(input, weights, axis=-1, output=None, mode="reflect",
                cval=0.0, origin=0):
    
    input = np.asarray(input)
    weights = np.asarray(weights)
    complex_input = input.dtype.kind == 'c'
    complex_weights = weights.dtype.kind == 'c'
    if complex_input or complex_weights:
        if complex_weights:
            weights = weights.conj()
            weights = weights.astype(np.complex128, copy=False)
        kwargs = dict(axis=axis, mode=mode, origin=origin)
        output = _ni_support._get_output(output, input, complex_output=True)
        return _complex_via_real_components(correlate1d, input, weights,
                                            output, cval, **kwargs)

    output = _ni_support._get_output(output, input)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] < 1:
        raise RuntimeError('no filter weights given')
    if not weights.flags.contiguous:
        weights = weights.copy()
    axis = normalize_axis_index(axis, input.ndim)
    if _invalid_origin(origin, len(weights)):
        raise ValueError('Invalid origin; origin must satisfy '
                         '-(len(weights) // 2) <= origin <= '
                         '(len(weights)-1) // 2')
    mode = _ni_support._extend_mode_to_code(mode)
    _nd_image.correlate1d(input, weights, axis, output, mode, cval,
                          origin)
    return output


def _wallmodel_kernel1d(radius, avespa, order):
    #print(radius, order)
    wallmodel = {"1-1":[11, 0, -0.616184226, 0, 0.015039424, 0, -0.000140206, 0, 4.42577e-7, 0],
                "1-0":[11, 0, 0, -0.308092113, 0, 0.003759856, 0, -2.33677e-5, 0, 5.53222e-8],
                "2-1":[18, 0, -0.568900032, -2.66996e-15, 0.005932413, 1.5315e-17, -2.15327e-5, -2.54519e-20, 2.61033e-8, 0],
                "2-0":[18, 0, 0, -0.284450016, -8.89986e-16, 0.001483103, 3.063e-18, -3.58879e-6, -3.63598e-21, 3.26291e-9],
                "3-1":[28, 0, -0.663730928, -5.77742e-16, 0.002449185, 1.35283e-18, -3.62504e-6, -9.24775e-22, 1.83489e-9, 0],
                "3-0":[28, 0, 0, -0.331865464, -1.92581e-16, 0.000612296, 2.70567e-19, -6.04174e-7, -1.32111e-22, 2.29361e-10],
                "4-1":[29, 0, -0.633880075, 0, 0.002231013, 0, -2.9913e-6, 0, 1.36612e-9, 0],
                "4-0":[29, 0, 0, -0.316940037, 0, 0.000557753, 0, -4.9855e-7, 0, 1.70765E-10],
                "5-1":[30, 0, -0.601054448, 0, 0.002095178, 0, -2.64164e-6, 0, 1.12889e-9, 0],
                "5-0":[30, 0, 0, -0.300527224, 0, 0.000523795, 0, -4.40274e-7, 0, 1.41111e-10],
                "7-1":[34, 0, -0.621391637, -4.25916e-16, 0.001601946, 6.71785e-19, -1.5745e-6, -3.08094e-22, 5.23129e-10, 0],
                "7-0":[34, 0, 0, -0.310695819, -1.41972e-16, 0.000400487, 1.34357e-19, -2.62417e-7, -4.40134e-23, 6.53912e-11],
                "9-1":[48, 0, -0.645146943, 0, 0.000795406, 0, -3.95962e-7, 0, 6.74645e-11, 0],
                "9-0":[48, 0, 0, -0.322573472, 0, 0.000198851, 0, -6.59937e-8, 0, 8.43306e-12],
                "11-1":[52, 0, -0.606833389, 0, 0.0006452, 0, -2.60062e-7, 0, 3.58778e-11, 0],
                "11-0":[52, 0, 0, -0.303416695, 0, 0.0001613, 0, -4.33437e-8, 0, 4.48473e-12],
                "14-1":[43, 0, -0.611305227, -7.41751e-16, 0.000948135, 7.35023e-19, -5.85277e-7, -2.11966e-22, 1.22542e-10, 0],
                "14-0":[43, 0, 0, -0.305652614, -2.4725e-16, 0.000237034, 1.47005e-19, -9.75461e-8, -3.02809e-23, 1.53178e-11],
                "17-1":[39, 0, -0.599260755, 0, 0.001129757, 0, -8.45242e-7, 0, 2.14969e-10, 0],
                "17-0":[39, 0, 0, -0.299630377, 0, 0.000282439, 0, -1.40874e-7, 0, 2.68711e-11],
                "20-1":[45, 0, -0.561094326, 0, 0.000790389, 0, -4.41724e-7, 0, 8.42742e-11, 0],
                "20-0":[45, 0, 0, -0.280547163, 0, 0.000197597, 0, -7.36207e-8, 0, 1.05343e-11], 
                "25-1":[19, 0, -0.527983576, 0, 0.004564197, 0, -1.34278e-5, 0, 1.31231e-8, 0],
                "25-0":[19, 0, 0, -0.263991788, 0, 0.001141049, 0, -2.23796e-6, 0, 1.64038e-9],
                "30-1":[19, 0, -0.631904746, 1.38962e-15, 0.005142876, -6.879e-18, -1.52655e-5, 9.89426e-21, 1.54238e-8, 0],
                "30-0":[19, 0, 0, -0.315952373, 4.63207e-16, 0.001285719, -1.3758e-18, -2.54425e-6, 1.41347e-21, 1.92798e-9]}
    
    """
    血管の輝度変化を考慮したカーネルを計算する。
    """
    
    coefficient_name = str(int(radius)) + "-" + str(int(order)) 
    coefficient = wallmodel[coefficient_name]

    x_edge = coefficient[0]     #カーネルの端のx座標
    r_voxel = radius/avespa       #抽出したい半径が何ボクセル分か
    step = x_edge/r_voxel         #カーネルのx座用の刻み幅
    kernel = np.empty(0)
    for x in np.arange(-x_edge, x_edge, step):
        y = coefficient[1] + coefficient[2]*x + coefficient[3]*x**2 + coefficient[4]*x**3 + coefficient[5]*x**4 + coefficient[6]*x**5 + coefficient[7]*x**6 + coefficient[8]*x**7 + coefficient[9]*x**8      
        kernel = np.append(kernel, y)
    
    # if np.sum(kernel) != 0:
    #     kernel = kernel / np.sum(kernel)
    kernel = kernel / np.abs(kernel).max()
    
    # plt.plot(kernel)
    # plt.savefig(f"d:\\takahashi_k\\new_function\\wallmodel_kernel\\radius({radius})_order({order}).png")
    # plt.clf()

    return kernel

@_ni_docstrings.docfiller
def wallmodel_filter1d(input, radius, spa, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    
    # correlateを呼び出しているのであって、convolveを呼び出しているわけではないので、カーネルを元に戻す。
    weights = _wallmodel_kernel1d(radius, spa, order)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)         #畳み込みを計算


@_ni_docstrings.docfiller
def wallmodel_filter(input, radius, spa, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0):
    
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)                 #入力画像と同じサイズのゼロで初期化された配列を返す
    orders = _ni_support._normalize_sequence(order, input.ndim)     #入力が１つのスカラーの場合、入力画像の次数に等しい長さの配列を作成する。入力がシーケンスの場合、その長さが入力画像の次元数と等しいかどうかをチェックする。
    radiuses = _ni_support._normalize_sequence(radius, input.ndim)
    spas = _ni_support._normalize_sequence(spa, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    axes = list(range(input.ndim))                                  #[0, 1, 2]
    axes = [(axes[ii], radiuses[ii], spas[ii], orders[ii], modes[ii])
            for ii in range(len(axes))]
    if len(axes) > 0:
        for axis, radius, spa, order, mode in axes:
            wallmodel_filter1d(input, radius, spa, axis, order, output,
                              mode, cval, truncate)
            input = output
    else:
        output[...] = input[...]
    return output

def _hessian_matrix_with_wallmodel(image, radius, spa, mode='reflect', cval=0,
                                  order='rc'):
    
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == "xy":
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ["rc", "xy"]:
        raise ValueError(f"unrecognized order: {order}")

    # if np.isscalar(radius):
    #     radius = (radius,) * image.ndim

    # common_kwargs = dict(parameter, mode=mode, cval=cval)
    # gaussian_ = functools.partial(sigmoid2_filter, **common_kwargs)             #部分適応のためのコード

    # 2つの連続した1次ガウス微分演算を適用する。
    # 詳しくは以下のサイト:
    # https://dsp.stackexchange.com/questions/78280/are-scipy-second-order-gaussian-derivatives-correct  # noqa

    # 1.) 一方の軸は1次、他方の軸は平滑化（次数=0）
    ndim = image.ndim               #ndim = 配列の次元数　= 3

    # orders in 2D = ([1, 0], [0, 1])
    #        in 3D = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    #        etc.
    orders = tuple([0] * d + [1] + [0]*(ndim - d - 1) for d in range(ndim))
    gradients = [wallmodel_filter(image, radius, spa, order=orders[d], mode=mode, cval=cval) for d in range(ndim)]            #gaussian_filterに引数を渡す

    # 2.) 別の軸にも微分を適用する
    axes = range(ndim)                                                              #axes = 0 1 2
    if order == 'xy':
        axes = reversed(axes)
    H_elems = [wallmodel_filter(gradients[ax0], radius, spa, order=orders[ax1], mode=mode, cval=cval)                         #gaussian_filterに引数を渡す
               for ax0, ax1 in combinations_with_replacement(axes, 2)]              #(ax0 ax1) = (0 0), (0 1), (0 2), (1 1), (1 2), (2 2)
    return H_elems

def hessian_matrix(image, radius, spa, mode='constant', cval=0, order='rc'):


    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == "xy":
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ["rc", "xy"]:
        raise ValueError(f"unrecognized order: {order}")

    return _hessian_matrix_with_wallmodel(image, radius, spa, mode=mode,
                                             cval=cval, order=order)

def _symmetric_compute_eigenvalues(S_elems):
    

    if len(S_elems) == 3:  # Fast explicit formulas for 2D.
        M00, M01, M11 = S_elems
        eigs = np.empty((2, *M00.shape), M00.dtype)
        eigs[:] = (M00 + M11) / 2
        hsqrtdet = np.sqrt(M01 ** 2 + ((M00 - M11) / 2) ** 2)
        eigs[0] += hsqrtdet
        eigs[1] -= hsqrtdet
        return eigs
    else:
        matrices = _symmetric_image(S_elems)
        # eigvalsh returns eigenvalues in increasing order. We want decreasing
        eigs = np.linalg.eigvalsh(matrices)[..., ::-1]
        leading_axes = tuple(range(eigs.ndim - 1))
        return np.transpose(eigs, (eigs.ndim - 1,) + leading_axes)


def _symmetric_image(S_elems):
    
    image = S_elems[0]
    symmetric_image = np.zeros(image.shape + (image.ndim, image.ndim),
                               dtype=S_elems[0].dtype)
    for idx, (row, col) in \
            enumerate(combinations_with_replacement(range(image.ndim), 2)):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image

def hessian_matrix_eigvals(H_elems):
    
    return _symmetric_compute_eigenvalues(H_elems)


def frangi(image, radiuses, spa, alpha=0.5, beta=0.5, gamma=None,
           mode='reflect', cval=0):

    #image = -image
    filtered_max = np.zeros_like(image)
    for radius in radiuses:  # Filter for all sigmas.
        print(f"radius:{radius}")                                #2023/8/30 TAKAHASHI.K
        #spa.insert(1, radius)
        eigvals = hessian_matrix_eigvals(hessian_matrix(
            image, radius, spa, mode=mode, cval=cval))
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
    return filtered_max  # Return pixel-wise max over all sigmas.

# if __name__ == '__main__':
#     array = cv2.imread("D:\\takahashi_k\\new_function\\2d\\BOAT.bmp", cv2.IMREAD_GRAYSCALE)
#     #array, spa, ori = vtk_data_loader("D:\\takahashi_k\\new_function\\simulation\\vessel\\v3\\vessel_v3_rev_pad_label_noise_shadow.vti") 
#     for i in range(1,21,1):
#         radiuses = range(i, i+1, 1)
#         thickness = 2
#         ratio = 0.3
#         alpha_sk = 10
#         parameter = [alpha_sk, thickness, ratio]
#         print("frangi")
#         output = np.zeros_like(array)
#         for rd in radiuses:
#             temp = frangi(array, radiuses=range(rd, rd+1, 1), parameter=parameter)
#             output = np.maximum(output, temp)
#         output = output * (255/np.max(output))

#         cv2.imwrite(f"D:\\takahashi_k\\new_function\\2d\\sigmoid2\\BOAT_sigmiod2({i},{i+1},1).bmp", output)
#         #output = numpy_to_vtk(output, spa, ori)
#         #save_vtk(output, f"D:\\takahashi_k\\new_function\\simulation\\vessel\\v3\\sigmoid\\sigmoid({i},{i+1},1).vti")

if __name__ == '__main__':
    In_path = r'd:\takahashi_k\new_function\simulation\vessel\v2'
    In_name = "vessel_v2_rev_pad_us_noise_shadow"
    In_root = os.path.join(In_path + "\\" + In_name + '.vti')
    array, spa, ori = vtk_data_loader(In_root) 
    avespa = sum(spa)/3

    #シングルスケール
    # radius_list = [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30]
    # #for root in rootlist:
    # output = np.zeros_like(array)
    # for r in radius_list:
    #     output = frangi(array, radiuses=range(r, r+1, 1), spa=avespa)
    #     output = output * (255/np.max(output))

    #     #出力
    #     filename = In_root.replace(In_path, "")
    #     filename = filename.replace(".vti", "")
    #     outpath = os.path.join(In_path + f"\\wallmodel\\spa({avespa})r(single)")
    #     if not os.path.exists(outpath):
    #         os.mkdir(outpath)
    #     outroot = os.path.join(outpath + f"\\wallmodel_r({r}).vti")
    #     output = numpy_to_vtk(output, spa, ori)
    #     save_vtk(output, outroot)
    
    #マルチスケール
    #radius_list = [1, 2, 3, 4, 5]
    #radius_list = [1, 2, 3, 4, 5, 7, 9, 11]
    #radius_list = [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20]
    radius_list = [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30]
    #for root in rootlist:
    output = np.zeros_like(array)
    for r in radius_list:
        temp = frangi(array, radiuses=range(r, r+1, 1), spa=avespa)
        output = np.maximum(output, temp)
    output = output * (255/np.max(output))

    #出力
    filename = In_root.replace(In_path, "")
    filename = filename.replace(".vti", "")
    # outpath = os.path.join(In_path + f"\\allmodel\\spa({avespa})r(single)")
    # if not os.path.exists(outpath):
    #     os.mkdir(outpath)
    outroot = os.path.join(In_path + f"\\wallmodel\\r(1-30).vti")
    output = numpy_to_vtk(output, spa, ori)
    save_vtk(output, outroot)

# if __name__ == '__main__':
#     inpath = r"d:\takahashi_k\new_function\us(slice)\wall"
#     rootlist = glob.glob(f"{inpath}/*.png")
    
#     #各スケール加算
#     radius_list = [2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30]
#     for root in rootlist:
#         array = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
#         output = np.zeros_like(array)
#         for r in radius_list:
#             temp = frangi(array, radiuses=range(r, r+1, 1), spa=1)
#             output = np.maximum(output, temp)
#         output = output * (255/np.max(output))

#         #出力
#         filename = root.replace(inpath, "")
#         filename = filename.replace(".png", "")
#         outroot = os.path.join(inpath + "\\wallmodel\\spa(1)r(2-30)" + filename + ".png")
#         print(outroot)
#         cv2.imwrite(outroot, output)

    #各スケールのみ
    # radius_list = [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30]
    # for root in rootlist:
    #     array = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
    #     output = np.zeros_like(array)
    #     for r in radius_list:
    #         output = frangi(array, radiuses=range(r, r+1, 1), spa=1)
    #         output = output * (255/np.max(output))

    #         #出力
    #         filename = root.replace(inpath, "")
    #         filename = filename.replace(".png", "")
    #         outroot = os.path.join(inpath + "\\wallmodel\\spa(1)r(step)" + filename + f"_r({r}).png")
    #         #print(outroot)
    #         cv2.imwrite(outroot, output)
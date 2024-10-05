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


def _allmodel_kernel1d(radius, avespa, order):
    #print(radius, order)
    allmodel = {"1-1":[11, 0, -0.841327981, -5.05766e-15, 0.010572478,7.40601e-17,-7.39078e-5,-3.2002e19,1.9229e-7, 0],
                "1-0":[11, 0, 0, -0.420663991, -1.68589e-15, 0.002643119, 1.4812e-17, -1.2318e-5, -4.57171e-20, 2.40363e-8],
                "2-1":[18, 0, -0.778023619, 0, 0.004453894, 0, -1.30075e-5, 0, 1.40059e-8, 0],
                "2-0":[18, 0, 0, -0.389011809, 0, 0.001113474, 0, -2.16791e-6, 0, 1.75074e-9],
                "3-1":[28, 0, -0.773049376, 0, 0.001951339, 0, -2.45079e-6, 0, 1.12423e-9, 0],
                "3-0":[28, 0, 0, -0.386524688, 0, 0.000487835, 0, -4.08464e-7, 0, 1.40529e-10],
                "4-1":[29, 0, -0.762910143, -9.29105e-16, 0.001742468, 2.00505e-18, -1.94789e-6, -1.26204e-21, 7.96364e-10, 0],
                "4-0":[29, 0, 0, -0.381455071, -3.09702e-16, 0.000435617, 4.0101e-19, -3.24649e-7, -1.80291e-22, 9.95454e-11],
                "5-1":[30, 0, -0.711865271, 0, 0.001672432, 0, -1.79666e-6, 0, 6.95948e-10, 0],
                "5-0":[30, 0, 0, -0.355932635, 0, 0.000418108, 0, -2.99444e-7, 0, 8.69935e-11],
                "7-1":[34, 0, -0.652582087, 2.49371E-17, 0.001481406, -4.68469e-20, -1.39117e-06, 2.34967e-23, 4.51822e-10, 0],
                "7-0":[34, 0, 0, -0.326291043, 8.31237E-18, 0.000370351, -9.36938E-21, -2.31862e-7, 3.35667e-24, 5.64777E-11],
                "9-1":[48, 0, -0.694490033, 4.40687e-16, 0.000687398, -3.48026E-19, -3.04409e-7, 8.00362e-23, 4.77859e-11, 0],
                "9-0":[48, 0, 0, -0.347245016, 1.46896e-16, 0.00017185, -6.96052e-20, -5.07348e-8, 1.14337e-23,	5.97324e-12],
                "11-1":[52, 0, -0.671359925, 0, 0.000469572, 0, -1.44417e-7, 0, 1.53643e-11, 0],
                "11-0":[52, 0, 0, -0.335679963, 0, 0.000117393, 0, -2.40695e-8, 0, 1.92054e-12],
                "14-1":[51, 0, -0.708495171, 0, 0.00053443, 0, -1.95838e-7, 0, 2.61167e-11, 0],
                "14-0":[51, 0, 0, -0.354247586, 0, 0.000133607, 0, -3.26396e-8, 0, 3.26458e-12],
                "17-1":[64, 0, -0.660107748, 0, 0.000277035, 0, -5.45468e-8, 0, 3.89717e-12, 0],
                "17-0":[64, 0, 0, -0.330053874, 0, 6.92587e-5, 0, -9.09113e-9, 0, 4.87146e-13],
                "20-1":[73, 0, -0.642565882, -2.66074e-16, 0.000249229, 9.04368e-20, -4.45768e-8, -8.96726e-24, 2.88828e-12, 0],
                "20-0":[73, 0, 0, -0.321282941, -8.86914e-17, 6.23073e-5, 1.80874E-20, -7.42947e-9, -1.28104e-24, 3.61035e-13], 
                "25-1":[19, 0, -0.567419498, -3.00022e-15, 0.004257496, 1.54243e-17, -1.17504e-5, -2.31244e-20, 1.07125e-8, 0],
                "25-0":[19, 0, 0, -0.283709749, -1.00007e-15, 0.001064374, 3.08486e-18, -1.95841e-6, -3.30348e-21, 1.33906E-9],
                "30-1":[21, 0, -0.688126709, 1.80744e-15, 0.004006578, -7.29971e-18, -9.53486e-6, 8.5786e-21, 7.93576e-9, 0],
                "30-0":[21, 0, 0, -0.344063354, 6.02479e-16, 0.001001645, -1.45994e-18, -1.58914e-6, 1.22551e-21, 9.9197e-10]}
    
    """
    血管の輝度変化を考慮したカーネルを計算する。
    """
    
    coefficient_name = str(int(radius)) + "-" + str(int(order)) 
    coefficient = allmodel[coefficient_name]

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
    # plt.savefig(f"d:\\takahashi_k\\new_function\\allmodel_kernel\\radius({radius})_order({order}).png")
    # plt.clf()

    return kernel

@_ni_docstrings.docfiller
def allmodel_filter1d(input, radius, spa, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0):
    
    # correlateを呼び出しているのであって、convolveを呼び出しているわけではないので、カーネルを元に戻す。
    weights = _allmodel_kernel1d(radius, spa, order)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)         #畳み込みを計算


@_ni_docstrings.docfiller
def allmodel_filter(input, radius, spa, order=0, output=None,
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
            allmodel_filter1d(input, radius, spa, axis, order, output,
                              mode, cval, truncate)
            input = output
    else:
        output[...] = input[...]
    return output

def _hessian_matrix_with_allmodel(image, radius, spa, mode='reflect', cval=0,
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
    gradients = [allmodel_filter(image, radius, spa, order=orders[d], mode=mode, cval=cval) for d in range(ndim)]            #gaussian_filterに引数を渡す

    # 2.) 別の軸にも微分を適用する
    axes = range(ndim)                                                              #axes = 0 1 2
    if order == 'xy':
        axes = reversed(axes)
    H_elems = [allmodel_filter(gradients[ax0], radius, spa, order=orders[ax1], mode=mode, cval=cval)                         #gaussian_filterに引数を渡す
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

    return _hessian_matrix_with_allmodel(image, radius, spa, mode=mode,
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
    #     outpath = os.path.join(In_path + f"\\allmodel\\spa({avespa})r(single)")
    #     if not os.path.exists(outpath):
    #         os.mkdir(outpath)
    #     outroot = os.path.join(outpath + f"\\allmodel_r({r}).vti")
    #     output = numpy_to_vtk(output, spa, ori)
    #     save_vtk(output, outroot)
    
    #マルチスケール
    #radius_list = [1, 2, 3, 4, 5]
    radius_list = [1, 2, 3, 4, 5, 7, 9, 11]
    #radius_list = [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20]
    #radius_list = [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30]
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
    outroot = os.path.join(In_path + f"\\allmodel\\r(1-5).vti")
    output = numpy_to_vtk(output, spa, ori)
    save_vtk(output, outroot)


# if __name__ == '__main__':
#     inpath = r"d:\takahashi_k\new_function\us(slice)\wall"
#     rootlist = glob.glob(f"{inpath}/*.png")

    # #各スケール加算
    # radius_list = [3, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30]
    # for root in rootlist:
    #     array = cv2.imread(root, cv2.IMREAD_GRAYSCALE)
    #     output = np.zeros_like(array)
    #     for r in radius_list:
    #         temp = frangi(array, radiuses=range(r, r+1, 1), spa=1)
    #         output = np.maximum(output, temp)
    #     output = output * (255/np.max(output))

    #     #出力
    #     filename = root.replace(inpath, "")
    #     filename = filename.replace(".png", "")
    #     outroot = os.path.join(inpath + "\\allmodel\\spa(1)r(3-30)" + filename + ".png")
    #     print(outroot)
    #     cv2.imwrite(outroot, output)

    # 各スケールのみ
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
    #         outroot = os.path.join(inpath + "\\allmodel\\spa(1)r(single)" + filename + f"_r({r}).png")
    #         #print(outroot)
    #         cv2.imwrite(outroot, output)
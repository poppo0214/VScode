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

import cv2
import matplotlib.pyplot as plt
import glob
import os

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
    """与えられた軸に沿って1次元相関を計算する。

    与えられた軸に沿った配列の線を、与えられた重みで相関させる。

    Parameters
    ----------
    %(input)s
    weights : array
        1-D sequence of numbers.
    %(axis)s
    %(output)s
    %(mode_reflect)s
    %(cval)s
    %(origin)s

    Examples
    --------
    >>> from scipy.ndimage import correlate1d
    >>> correlate1d([2, 8, 0, 4, 1, 9, 9, 0], weights=[1, 3])
    array([ 8, 26,  8, 12,  7, 28, 36,  9])
    """
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
    
def _gaussian_kernel1d(sigma, order, radius):
    """
    1次元ガウシアン畳み込みカーネルを計算する。
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)               # exponent_range = [0 ,1]
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)              #ガウス関数
    phi_x = phi_x / phi_x.sum()                         #正規化

    if order == 0:
        #print(phi_x)
        # plt.plot(phi_x)
        # plt.savefig(f"D:\\takahashi_k\\new_function\\2d\\gaussian_kernel\\sigma({sigma})_order({order})_radius({radius}).png")
        # plt.clf()
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)                  # q = [0, 0]
        q[0] = 1                                 # q = [1, 0]      
        D = np.diag(exponent_range[1:], 1)       # D @ q(x) = q'(x)　diag：x座標が対角成分の対角行列を生成
        P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
        Q_deriv = D + P
        for _ in range(order):
            q = Q_deriv.dot(q)
        q = (x[:, None] ** exponent_range).dot(q)
        #print(q * phi_x)
        # plt.plot(q * phi_x)
        # plt.savefig(f"D:\\takahashi_k\\new_function\\2d\\gaussian_kernel\\sigma({sigma})_order({order})_radius({radius}).png")
        # plt.clf()
        return q * phi_x

@_ni_docstrings.docfiller
def gaussian_filter1d(input, sigma, axis=-1, order=0, output=None,
                      mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """1次元ガウシアンフィルタ。

    Parameters
    ----------
    %(input)s
    sigma : scalar
        ガウシアンカーネルの標準偏差
    %(axis)s
    order : int, optional
        次数0は、ガウスカーネルとの畳み込みに対応する。
        正の次数は、ガウスの微分との畳み込みに対応する。
    %(output)s
    %(mode_reflect)s
    %(cval)s
    truncate : float, optional
        標準偏差の数でフィルタを切り捨てる。
        デフォルトは4.0。
    radius : None or int, optional
        ガウスカーネルの半径。指定した場合、カーネルのサイズは ``2*radius + 1`` となり、 
        `truncate` は無視される。
        デフォルトは None。

    Returns
    -------
    gaussian_filter1d : ndarray

    Notes
    -----
    ガウシアンカーネルのサイズは ``2*radius + 1`` で、
    ここで ``radius = round(truncate * sigma)`` です。

    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter1d
    >>> import numpy as np
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
    array([ 1.42704095,  2.06782203,  3.        ,  3.93217797,  4.57295905])
    >>> gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    array([ 2.91948343,  2.95023502,  3.        ,  3.04976498,  3.08051657])
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.standard_normal(101).cumsum()
    >>> y3 = gaussian_filter1d(x, 3)
    >>> y6 = gaussian_filter1d(x, 6)
    >>> plt.plot(x, 'k', label='original data')
    >>> plt.plot(y3, '--', label='filtered, sigma=3')
    >>> plt.plot(y6, ':', label='filtered, sigma=6')
    >>> plt.legend()
    >>> plt.grid()
    >>> plt.show()

    """
    sd = float(sigma)
    # フィルタの半径を標準偏差の切り捨てに等しくする。
    lw = int(truncate * sd + 0.5)               
    if radius is not None:
        lw = radius
    if not isinstance(lw, numbers.Integral) or lw < 0:
        raise ValueError('Radius must be a nonnegative integer.')
    # correlateを呼び出しているのであって、convolveを呼び出しているわけではないので、カーネルを元に戻す。
    weights = _gaussian_kernel1d(sigma, order, lw)[::-1]
    return correlate1d(input, weights, axis, output, mode, cval, 0)         #畳み込みを計算


@_ni_docstrings.docfiller
def gaussian_filter(input, sigma, order=0, output=None,
                    mode="reflect", cval=0.0, truncate=4.0, *, radius=None):
    """多次元ガウシアンフィルタ。

    Parameters
    ----------
    %(input)s
    sigma : scalar or sequence of scalars
        ガウシアンカーネルの標準偏差。
        ガウシアンフィルタの標準偏差は,各軸ごとにシーケンスとして与えられるか,1つの数値として与えられ,
        その場合はすべての軸について等しい．
    order : int or sequence of ints, optional
        各軸に沿ったフィルタの次数は整数のシーケンス、または1つの数値として与えられる。
        次数0はガウスカーネルとの畳み込みに対応する。
        正の次数はガウスの微分との畳み込みに対応する.
        orders in 3D = [1, 0, 0]->[0, 1, 0]->[0, 0, 1]
    %(output)s
    %(mode_multiple)s
    %(cval)s
    truncate : float, optional
        標準偏差の数でフィルタを切り捨てる。
        デフォルトは4.0。
    radius : None or int or sequence of ints, optional
        ガウスカーネルの半径．
        半径は各軸ごとに連続した値として与えるか、単一の値として与える。
        指定された場合、各軸に沿ったカーネルのサイズは ``2*radius + 1`` となり、 
        `truncate` は無視されます。
        デフォルトはなし。

    Returns
    -------
    gaussian_filter : ndarray
        入力と同じ形の配列を返す

    Notes
    -----
    多次元フィルタは1次元畳み込みフィルタのシーケンスとして実装される。
    中間配列は出力と同じデータ型に格納される。
    したがって，精度が制限された出力型では，中間結果が不十分な精度で格納される可能性があるため，
    結果が不正確になる可能性がある．

    ガウシアンカーネルの各軸のサイズは ``2*radius + 1`` であり， 
    ``radius = round(truncate * sigma)`` となります．


    Examples
    --------
    >>> from scipy.ndimage import gaussian_filter
    >>> import numpy as np
    >>> a = np.arange(50, step=2).reshape((5,5))
    >>> a
    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28],
           [30, 32, 34, 36, 38],
           [40, 42, 44, 46, 48]])
    >>> gaussian_filter(a, sigma=1)
    array([[ 4,  6,  8,  9, 11],
           [10, 12, 14, 15, 17],
           [20, 22, 24, 25, 27],
           [29, 31, 33, 34, 36],
           [35, 37, 39, 40, 42]])

    >>> from scipy import datasets
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = datasets.ascent()
    >>> result = gaussian_filter(ascent, sigma=5)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)
    output = _ni_support._get_output(output, input)                 #入力画像と同じサイズのゼロで初期化された配列を返す
    orders = _ni_support._normalize_sequence(order, input.ndim)     #入力が１つのスカラーの場合、入力画像の次数に等しい長さの配列を作成する。入力がシーケンスの場合、その長さが入力画像の次元数と等しいかどうかをチェックする。
    sigmas = _ni_support._normalize_sequence(sigma, input.ndim)
    modes = _ni_support._normalize_sequence(mode, input.ndim)
    radiuses = _ni_support._normalize_sequence(radius, input.ndim)
    axes = list(range(input.ndim))                                  #[0, 1, 2]
    axes = [(axes[ii], sigmas[ii], orders[ii], modes[ii], radiuses[ii])
            for ii in range(len(axes)) if sigmas[ii] > 1e-15]
    if len(axes) > 0:
        for axis, sigma, order, mode, radius in axes:
            gaussian_filter1d(input, sigma, axis, order, output,
                              mode, cval, truncate, radius=radius)
            input = output
    else:
        output[...] = input[...]
    return output

def _hessian_matrix_with_gaussian(image, sigma=1, mode='reflect', cval=0,
                                  order='rc'):
    """ガウス導関数との畳み込みによってヘッセを計算する。

    2次元では、ヘッセ行列は次のように定義される。:
        H = [Hrr Hrc]
            [Hrc Hcc]

    これは画像とr方向とc方向のそれぞれのガウスカーネルの2次導関数の畳み込みによって計算される.
    ここでの実装はn次元データもサポートしている。

    Parameters
    ----------
    image : ndarray
        入力画像
    sigma : float or sequence of float, optional
        ガウシアンカーネルに使用される標準偏差。ピクセル距離で平滑化の量を設定する。シグマは 1.0よりはるかに小さいシグマを選択しないことをお勧めします。エイリアシング・アーティファクトが発生する可能性があります。
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        画像の枠外の値の扱い方。
    cval : float, optional
        mode'constant'と共に使用される、画像境界の外側の値。
    order : {'rc', 'xy'}, optional
        このパラメータは、グラデーション計算で画像軸の逆順または順順を使用することを可能にします。
        「rc」は最初の軸（Hrr, Hrc, Hcc）を最初に使用することを示し、「xy」は最後の軸（Hxx, Hxy, Hyy）を最初に使用することを示す。

    Returns
    -------
    H_elems : list of ndarray
        入力画像の各ピクセルのヘッセ行列の上対角要素。の上対角要素。
        2次元の場合、これは [Hrr、 Hrc, Hcc] を含む3要素のリストになる。
        nD の場合，このリストには ``(n**2 + n) / 2`` 個の配列が含まれます．

    """
    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == "xy":
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ["rc", "xy"]:
        raise ValueError(f"unrecognized order: {order}")

    if np.isscalar(sigma):
        sigma = (sigma,) * image.ndim

    """
    この関数は、`scipy.ndimage.gaussian_filter`を使用し、次数引数でコンボリューションを計算します。
    例えば， ``order=[1, 0]`` を指定すると、1番目の軸に沿ってガウスの1階微分による畳み込みが適用され、
    2番目の軸に沿って単純なガウス平滑化が適用されます。

    シグマが小さい場合、SciPyのガウシアンフィルタはエイリアシングやエッジのアーチファクトに悩まされます。
    これは、フィルタが非常にゆっくりとしか0にならない（次数1/n**2）sincまたはsinc微分を近似するためです。
    したがって、エッジのアーチファクトを減らすために、より大きな切り捨て値を使用します。
    """
    
    truncate = 8 if all(s > 1 for s in sigma) else 100
    sq1_2 = 1 / math.sqrt(2)
    sigma_scaled = tuple(sq1_2 * s for s in sigma)
    common_kwargs = dict(sigma=sigma_scaled, mode=mode, cval=cval,
                         truncate=truncate)
    gaussian_ = functools.partial(gaussian_filter, **common_kwargs)             #部分適応のためのコード

    # 2つの連続した1次ガウス微分演算を適用する。
    # 詳しくは以下のサイト:
    # https://dsp.stackexchange.com/questions/78280/are-scipy-second-order-gaussian-derivatives-correct  # noqa

    # 1.) 一方の軸は1次、他方の軸は平滑化（次数=0）
    ndim = image.ndim               #ndim = 配列の次元数　= 3

    # orders in 2D = ([1, 0], [0, 1])
    #        in 3D = ([1, 0, 0], [0, 1, 0], [0, 0, 1])
    #        etc.
    orders = tuple([0] * d + [1] + [0]*(ndim - d - 1) for d in range(ndim))
    gradients = [gaussian_(image, order=orders[d]) for d in range(ndim)]            #gaussian_filterに引数を渡す

    # 2.) 別の軸にも微分を適用する
    axes = range(ndim)                                                              #axes = 0 1 2
    if order == 'xy':
        axes = reversed(axes)
    H_elems = [gaussian_(gradients[ax0], order=orders[ax1])                         #gaussian_filterに引数を渡す
               for ax0, ax1 in combinations_with_replacement(axes, 2)]              #(ax0 ax1) = (0 0), (0 1), (0 2), (1 1), (1 2), (2 2)
    return H_elems

def hessian_matrix(image, sigma=1, mode='constant', cval=0, order='rc',
                   use_gaussian_derivatives=None):
    r"""ヘッセ行列を計算する。

    2Dでは、ヘッセ行列は次のように定義される::

        H = [Hrr Hrc]
            [Hrc Hcc]

    この行列は、画像をそれぞれのrおよびc方向におけるガウシアンカーネルの2次導関数で画像を畳み込むことによって計算される。
    ここでの実装はn次元データもサポートしている。

    Parameters
    ----------
    image : ndarray
        入力画像.
    sigma : float
        自己相関行列の重み付け関数として使用されるガウシアンカーネルに使用される標準偏差。
        自己相関行列の重み付け関数として使用される。
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        画像の枠外の値の扱い方。
    cval : float, optional
        mode'constant'と共に使用される、画像境界の外側の値。
    order : {'rc', 'xy'}, optional
        2D画像の場合、このパラメータによって、グラデーション計算で画像軸の逆順または順順を使用することができる。
        「rc」は最初の軸（Hrr, Hrc, Hcc）の使用を示し、「xy」は最後の軸（Hxx, Hxy, Hyy）の使用を示します。
        高次元の画像は常に'rc'順序を使用しなければならない。
    use_gaussian_derivatives : boolean, optional
        ヘッセがガウス導関数との畳み込みによって計算されるか，単純な有限差分演算によって計算されるかを示す．
        導関数で畳み込むか, あるいは単純な有限差分演算で計算するかを指定する.

    Returns
    -------
    H_elems : list of ndarray
        入力画像の各ピクセルに対するヘシアンマトリクスの上対角要素。
        2次元の場合、これは [Hrr、 Hrc, Hcc] を含む3要素のリストになります。
        nD の場合，このリストには ``(n**2 + n) / 2`` 個の配列が含まれます．


    Notes
    -----
    微分と畳み込みの分配特性により、ガウスカーネルGで平滑化された画像Iの微分を、
    画像とGの微分の畳み込みとして言い直すことができる。
    .. math::

        \frac{\partial }{\partial x_i}(I * G) =
        I * \left( \frac{\partial }{\partial x_i} G \right)

    use_gaussian_derivatives`` が ``True`` の場合、このプロパティはヘッセ行列を構成する2次導関数の計算に使用されます。

    use_gaussian_derivatives`` が ``False`` の場合、ガウス平滑化された行列に対して単純な有限差分を計算する。
    ``False`` の場合、ガウス平滑化された画像に対する単純な差分が代わりに使用されます。

    Examples
    --------
    >>> from skimage.feature import hessian_matrix
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc',
    ...                                use_gaussian_derivatives=False)
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

    """

    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)
    if image.ndim > 2 and order == "xy":
        raise ValueError("order='xy' is only supported for 2D images.")
    if order not in ["rc", "xy"]:
        raise ValueError(f"unrecognized order: {order}")

    if use_gaussian_derivatives is None:
        use_gaussian_derivatives = False
        warn("use_gaussian_derivatives currently defaults to False, but will "
             "change to True in a future version. Please specify this "
             "argument explicitly to maintain the current behavior",
             category=FutureWarning, stacklevel=2)

    if use_gaussian_derivatives:
        return _hessian_matrix_with_gaussian(image, sigma=sigma, mode=mode,
                                             cval=cval, order=order)

    gaussian_filtered = gaussian(image, sigma=sigma, mode=mode, cval=cval)

    gradients = np.gradient(gaussian_filtered)
    axes = range(image.ndim)

    if order == 'xy':
        axes = reversed(axes)

    H_elems = [np.gradient(gradients[ax0], axis=ax1)
               for ax0, ax1 in combinations_with_replacement(axes, 2)]
    return H_elems

def _symmetric_compute_eigenvalues(S_elems):
    """対称行列の上対角の項目から固有値を計算する。

    Parameters
    ----------
    S_elems : list of ndarray
        hessian_matrix` または `structure_tensor` が返す行列の上対角要素。

    Returns
    -------
    eigs : ndarray
        行列の固有値を小さい順に並べたもの。固有値は先頭の次元である．
        つまり， ``eigs[i, j, k]`` は (j, k) の位置にある i 番目の固有値を含みます．
    """

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
    """行列の上対角要素を完全な対称行列に変換する。

    Parameters
    ----------
    S_elems : list of array
        hessian_matrix` または `structure_tensor` が返す行列の上対角要素。

    Returns
    -------
    image : array
        各座標に対応する行列を含む ``(M, N[, ...], image.ndim, image.ndim)`` 形式の配列．
    """
    image = S_elems[0]
    symmetric_image = np.zeros(image.shape + (image.ndim, image.ndim),
                               dtype=S_elems[0].dtype)
    for idx, (row, col) in \
            enumerate(combinations_with_replacement(range(image.ndim), 2)):
        symmetric_image[..., row, col] = S_elems[idx]
        symmetric_image[..., col, row] = S_elems[idx]
    return symmetric_image

def hessian_matrix_eigvals(H_elems):
    """ヘッセ行列の固有値を計算する。

    Parameters
    ----------
    H_elems : list of ndarray
        hessian_matrix` が返すヘッセ行列の上対角要素．

    Returns
    -------
    eigs : ndarray
        ヘッセ行列の固有値を小さい順に並べたもの．
        固有値は先頭の次元である．
        つまり, ``eigs[i, j, k]`` は, (j, k)の位置で最大の固有値を含みます.

    Examples
    --------
    >>> from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> H_elems = hessian_matrix(square, sigma=0.1, order='rc',
    ...                          use_gaussian_derivatives=False)
    >>> hessian_matrix_eigvals(H_elems)[0]
    array([[ 0.,  0.,  2.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 2.,  0., -2.,  0.,  2.],
           [ 0.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  2.,  0.,  0.]])
    """
    return _symmetric_compute_eigenvalues(H_elems)


def frangi(image, sigmas=range(1, 10, 2), scale_range=None,
           scale_step=None, alpha=0.5, beta=0.5, gamma=None,
           black_ridges=True, mode='reflect', cval=0):
    """
    フランジ血管フィルタ... フィルターを使って画像にフィルタをかける。
    このフィルターは血管やしわや川などの連続した隆起を検出するのに使えます。
    画像全体に占めるそのようなオブジェクトの割合を計算するのにも使えます。
    2次元画像と3次元画像に対してのみ定義されています。
    ヘシアンの固有ベクトルを計算し、[1]で述べられている方法に従って，画像領域と血管の類似度を計算する．

    Parameters
    ----------
    image : (N, M[, P]) ndarray
        入力画像データの配列。
    sigmas : iterable of floats, optional
        フィルタのスケールとして使用されるシグマ, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        σの範囲として使われる.
    scale_step : float, optional
        シグマ間のステップサイズ.
    alpha : float, optional
        フィルタの板状構造からのずれを調整するフランジ補正定数。板状構造からの逸脱に対する感度を調整する。
    beta : float, optional
        フランジ補正定数。塊状構造からの逸脱に対する感度を調整する。
    gamma : float, optional
        フィルタの感度を調整するフランジ補正定数。の感度を調整する。
        デフォルトの None は最大ヘシアン・ノルムの半分を使います。
    black_ridges : boolean, optional
        True (初期設定値) のときはフィルターは黒い稜線を検出します。偽のときは白い稜線を検出します。
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        画像の枠外の値の扱い方。
    cval : float, optional
        mode 'constant'と併用し、画像境界の外側の値を指定する。

    Returns
    -------
    out : (N, M[, P]) ndarray
        Filtered image (maximum of pixels across all scales).
    """
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
    print("sigmas = ", sigmas)                                 #2023/8/30 TAKAHASHI.K
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:  # Filter for all sigmas.
        print(f"sigma={sigma}")                                #2023/8/30 TAKAHASHI.K
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
    return filtered_max  # Return pixel-wise max over all sigmas.

# if __name__ == '__main__':
#     sigmas = range(20, 21, 1)
#     array = cv2.imread("D:\\takahashi_k\\new_function\\2d\\BOAT.bmp", cv2.IMREAD_GRAYSCALE)

#     #frangifilter
#     output = np.zeros_like(array)
#     for sigma in sigmas:
#         temp = frangi(array, sigmas=range(sigma, sigma+1, 1), black_ridges=True)
#         output = np.maximum(output, temp)
#     output = output*255

#     #出力
#     cv2.imwrite(f"D:\\takahashi_k\\new_function\\2d\\BOAT_frg(20,21,1).bmp", output)

if __name__ == '__main__':
    inpath = r"d:\takahashi_k\new_function\us(slice)\both"
    rootlist = glob.glob(f"{inpath}/*.png")
    sigmas = range(1, 11, 1)

    for root in rootlist:
        array = cv2.imread(root, cv2.IMREAD_GRAYSCALE)

        #frangifilter
        output = np.zeros_like(array)
        for sigma in sigmas:
            temp = frangi(array, sigmas=range(sigma, sigma+1, 1), black_ridges=False)
            output = np.maximum(output, temp)
        output = output*255

        #出力
        filename = root.replace(inpath, "")
        filename = filename.replace(".png", "")
        outroot = os.path.join(inpath + "\\frg_reverse" + filename + "_frg(1,11,1).png")
        #print(outroot)
        cv2.imwrite(outroot, output)
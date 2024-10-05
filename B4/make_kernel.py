import numpy as np
import sympy as sp
import cupy as cp
import math
import matplotlib.pyplot as plt

def _gaussian_kernel1d(sigma, order, radius):
    """
    1次元ガウシアン畳み込みカーネルを計算する。
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    exponent_range = np.arange(order + 1)               # exponent_range = [0, 1]
    print(f"exponent_range:{exponent_range}")
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    print(f"x:{x}")
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)              #ガウス関数
    #print(f"phi_x:{phi_x}")
    phi_x = phi_x / phi_x.sum()                         #正規化
    print(f"phi_x:{phi_x}")

    if order == 0:
        return phi_x
    else:
        # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
        # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
        # p'(x) = -1 / sigma ** 2
        # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
        # coefficients of q(x)
        q = np.zeros(order + 1)                     # q = [0, 0]
        print(f"q:{q}")
        q[0] = 1                                    # q = [1, 0]
        print(f"q:{q}")      
        D = np.diag(exponent_range[1:], 1)          # D @ q(x) = q'(x)　diag：exponent_rangeのスライスが対角成分の対角行列を生成
        print(f"D:{D}")
        P = np.diag(np.ones(order)/-sigma2, -1)     # P @ q(x) = q(x) * p'(x)
        print(f"P:{P}")
        Q_deriv = D + P
        print(f"Q_deriv:{Q_deriv}")                 #Q_deriv:微分の行列
        for _ in range(order):
            q = Q_deriv.dot(q)                      #Q_derivとqの内積 q->微分フィルタ？
            print(f"q:{q}")
        print(f"x[:, None] : {x[:, None]}")
        print(f"x[:, None] ** exponent_range : {x[:, None] ** exponent_range}")  
        q = (x[:, None] ** exponent_range).dot(q)   #x[:, None]：xの範囲を2次元配列にする　** exponent_range：[0 1]->[0乗 1乗]
        print(f"q:{q}")  
        output = q * phi_x
        print(f"output:{output}")  
        return output

def _step_kernel1d(radius, order):
    """
    1次元step畳み込みカーネルを計算する。
    """
    ex_range = radius*2             #血管外のボクセル数

    f1 = np.arange(1, ex_range+1)   #一階微分の際の血管外部分
    f2 = f1*f1/2                    #微分しない場合の血管外部分
    print(f"f1:{f1}")
    print(f"f2:{f2}")
    if order < 0 or order > 1:
        raise ValueError('order must be 0 or 1')

    elif order == 1:
        output = np.zeros(radius)
        output = np.append(output, f1)
        inverse = np.flipud(output)
        output = np.append(inverse, output)
        output = output / output.sum()  
        print(output)
    
    elif order == 0:
        output = np.zeros(radius)
        output = np.append(output, f2)
        inverse = np.flipud(output)
        output = np.append(inverse, output)
        output = output / output.sum()  
        print(output)
    
    return output

def _sigmoid_kernel1d(radius, order, alpha):
    """
    1次元シグモイド畳み込みカーネルを計算する。
    2階微分 : f=1/(1+exp(10(radius-x))
    alpha = 10              #血管境界部分の輝度勾配の大きさを決めるパラメータ、大きいほど輝度勾配が大きい
    radius = 5              #血管半径のボクセル数

    """
    kernel = radius*2+1

    if order < 0 or order > 1:
        raise ValueError('order must be 0 or 1')

    elif order == 1:
        x = np.arange(0, kernel, 1)
        f1 = np.log(np.exp(alpha*x) + np.exp(alpha*radius))/alpha
        f1 = f1 - math.log(math.exp(0) + math.exp(alpha*radius))/alpha
        inverse = np.flipud(f1)*(-1)
        output = np.append(inverse, f1)
        output = output / output.sum()  
        print(output)
    
    elif order == 0:
        f2 = np.empty(0)
        for i in range(0, kernel, 1):
            temp = radius*i - sp.polylog(2, -math.exp(-alpha*(radius-i))) / alpha*alpha
            f2 = np.append(f2, temp)

        inverse = np.flipud(f2)*(-1)
        output = np.append(inverse, f2)
        output = output / output.sum()  
        print(output)

    
    return output

def _sigmoid2_kernel1d(order, alpha, beta, gamma, delta=0.3):
    """
    血管壁を考慮した、1次元のシグモイド関数を2つ使った畳み込みカーネルを計算する。
    2階微分 : f=1/(1+exp(10(radius-x))
    alpha = 10              #血管境界部分の輝度勾配の大きさを決めるパラメータ、大きいほど輝度勾配が大きい
    beta : 内腔部分のsigmoid関数の変曲点
    gamma : 血管外のsigmoid関数の変曲点
    radius = beta-gamma
    delta = 0.3              #肝実質の輝度の割合 : 255*beta


    """
    
    radius = beta + (gamma-beta)/2
    kernel = radius*2
    radius = int(radius)

    if order < 0 or order > 1:
        raise ValueError('order must be 0 or 1')

    elif order == 1:
        f1 = np.empty(0)
        xin = np.arange(0, radius+1, 1)                                           #血管内腔の座標
        xout = np.arange(radius, kernel+1, 1)                                     #血管外の座標

        fin = np.log(np.exp(alpha*xin)+ np.exp(alpha*beta))/alpha               #血管内のsigmoid関数の1次積分
        fout = xout - (np.log(np.exp(alpha*xout)+ np.exp(alpha*gamma))/alpha) * delta       #血管外のsigmoid関数の1次微分
        diff = - fout[0] + fin[-1]                                   #血管内と血管外のsigmoid関数の接続
        fout = fout + diff
        fout = np.delete(fout, 0)
        f1 = np.append(fin, fout)
        f1 = f1 - f1[0]

        inverse = np.flipud(f1)*(-1)
        output = np.append(inverse, f1)
        inverse = np.delete(inverse, -1)
        output = output / output.sum()  
    
    elif order == 0:
        f2 = np.empty(0)
        for i in range(0, radius+1, 1):
            temp1 = beta*i - sp.polylog(2, -math.exp(alpha*(i-beta))) /( alpha*alpha)
            f2 = np.append(f2, temp1)

        f2 = np.delete(f2, -1)
        diff = -(radius*radius/2 - gamma*delta*radius + delta*sp.polylog(2, -math.exp(alpha*(radius-gamma))) / (alpha*alpha)) + temp1
        for j in range(radius, kernel+1, 1):
            exp = -math.exp(alpha*(j-gamma))
            temp2 = diff + j*j/2 - gamma*delta*j + delta*sp.polylog(2, exp) / (alpha*alpha)
            f2 = np.append(f2, temp2)

        inverse = np.flipud(f2)*(-1)
        inverse = np.delete(inverse, -1)
        output = np.append(inverse, f2)
        output = output / output.sum()  
        print(output)

    
    return output


def _allmodel_kernel1d(radius, order, avespa):
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
        #print(y)
        kernel = np.append(kernel, y)
    
    # if np.sum(kernel) != 0:
    #     kernel = kernel / np.sum(kernel)
    kernel = kernel / np.abs(kernel).max()

    
    plt.plot(kernel)
    plt.savefig(f"d:\\takahashi_k\\model\\analyze\\all\\kernel\\spa=1(NF=max)\\radius({radius})_order({order})_absmax({np.abs(kernel).max()}).png")
    plt.clf()

def _nonwallmodel_kernel1d(radius, order, avespa):
    nonwallmodel = {"1-1":[9, 0, -0.939739407, 2.73832e-15, 0.005070575, -2.1779e-17, -1.60099e-5, 5.05991e-20, 1.95016e-8, 0],
                "1-0":[9, 0, 0, -0.469869704, 9.12773e-16, 0.001267644, -4.35579e-18, -2.66832e-6, 7.22845e-21, 2.4377e-9],
                "2-1":[15, 0, -0.941503122, 0, 0.012769625, 0, -0.000102869, 0, 3.23934e-7, 0],
                "2-0":[15, 0, 0, -0.470751561, 0, 0.003192406, 0, -1.71448e-5, 0, 4.04917e-8],
                "3-1":[21, 0, -0.937572354, 3.7877e-16, 0.002429325, -1.53481e-18, -3.4448e-6, 1.81998e-21, 1.71311e-9, 0],
                "3-0":[21, 0, 0, -0.468786177, 1.26257e-16, 0.000607331, -3.06962e-19, -5.74133e-7, 2.59998e-22, 2.14139e-10],
                "4-1":[26, 0, -0.964377846, 0, 0.00131366, 0, -9.57922e-7, 0, 2.18031e-10, 0],
                "4-0":[26, 0, 0, -0.482188923, 0, 0.000328415, 0, -1.59654e-7, 0, 2.72538e-11],
                "5-1":[23, 0, -0.95002648, -4.54366e-16, 0.001685443, 5.91347e-19, -1.28723e-6, 0, 0, 0],
                "5-0":[23, 0, 0, -0.47501324, -1.51455E-16, 0.000421361, 1.18269e-19, -2.14538e-7, 0, 0],
                "7-1":[24, 0, -0.962336283, 2.01324e-16, 0.001594915, -6.1524e-19, -1.43666e-6, 5.49373e-22, 4.13949e-10, 0],
                "7-0":[24, 0, 0, -0.481168142, 6.7108e-17, 0.000398729, -1.23048e-19, -2.39444e-7, 7.84818e-23, 5.17436e-11],
                "9-1":[41, 0, -0.86634458, 0, 0.000627443, 0, -2.08262e-7, 0, 1.62811e-11, 0],
                "9-0":[41, 0, 0, -0.43317229, 0, 0.000156861, 0, -3.47104e-8, 0, 2.03514e-12],
                "11-1":[51, 0, -0.792318565, 0, 0.000288731, 0, -1.96961e-8, 0, -7.85503e-12, 0],
                "11-0":[51, 0, 0, -0.396159282, 0, 7.21827e-5, 0, -3.28269e-9, 0, -9.81879e-13],
                "14-1":[51, 0, -0.923088612, 1.26526e-16, 0.000338429, -3.4177e-20, -5.3505e-8, 0, 0, 0],
                "14-0":[51, 0, 0, -0.461544306, 4.21752e-17, 8.46072e-5, -6.8354e-21, -8.9175e-9, 0, 0],
                "17-1":[64, 0, -0.824678193, 0, 0.000164192, 0, -4.51105e-9, 0, -1.97107e-12, 0],
                "17-0":[64, 0, 0, -0.412339096, 0, 4.10479e-5, 0, -7.51842e-10, 0, -2.46383e-13],
                "20-1":[73, 0, -0.891505024, 0, 0.000193888, 0, -2.33277e-8, 0, 8.83688e-13, 0],
                "20-0":[73, 0, 0, -0.445752512, 0, 4.8472e-5, 0, -3.88795e-9, 0, 1.10461e-13], 
                "25-1":[5, 0, -0.932997046, 6.55835e-14, 0.028526724, -1.57373e-15, -0.000389602, 0, 0, 0],
                "25-0":[5, 0, 0, -0.466498523, 2.18612e-14, 0.00713168, -3.14746e-16, -6.49337e-5, 0, 0],
                "30-1":[21, 0, -1.084368676, 0, 0.002688113, 0, -4.46327e-6, 0, 3.04957e-9, 0],
                "30-0":[21, 0, 0, -0.542184338, 0, 0.000672028, 0, -7.43878E-7, 0, 3.81197e-10]}
    
    """
    血管壁がない血管の輝度変化を考慮したカーネルを計算する。
    """
    
    coefficient_name = str(int(radius)) + "-" + str(int(order)) 
    coefficient = nonwallmodel[coefficient_name]

    x_edge = coefficient[0]     #カーネルの端のx座標
    r_voxel = radius/avespa       #抽出したい半径が何ボクセル分か
    step = x_edge/r_voxel         #カーネルのx座用の刻み幅
    kernel = np.empty(0)
    for x in np.arange(-x_edge, x_edge, step):
        y = coefficient[1] + coefficient[2]*x + coefficient[3]*x**2 + coefficient[4]*x**3 + coefficient[5]*x**4 + coefficient[6]*x**5 + coefficient[7]*x**6 + coefficient[8]*x**7 + coefficient[9]*x**8      
        #print(y)
        kernel = np.append(kernel, y)
    
    # if np.sum(kernel) != 0:
    #     kernel = kernel / np.sum(kernel)
    kernel = kernel / np.abs(kernel).max()
    
    plt.plot(kernel)
    plt.savefig(f"d:\\takahashi_k\\model\\analyze\\nonwall\\kernel\\spa=1\\radius({radius})_order({order}).png")
    plt.clf()
        

def _wallmodel_kernel1d(radius, order, avespa):
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
    血管壁がある血管の輝度変化を考慮したカーネルを計算する。
    """
    
    coefficient_name = str(int(radius)) + "-" + str(int(order)) 
    coefficient = wallmodel[coefficient_name]

    x_edge = coefficient[0]     #カーネルの端のx座標
    r_voxel = radius/avespa       #抽出したい半径が何ボクセル分か
    step = x_edge/r_voxel         #カーネルのx座用の刻み幅
    kernel = np.empty(0)
    for x in np.arange(-x_edge, x_edge, step):
        y = coefficient[1] + coefficient[2]*x + coefficient[3]*x**2 + coefficient[4]*x**3 + coefficient[5]*x**4 + coefficient[6]*x**5 + coefficient[7]*x**6 + coefficient[8]*x**7 + coefficient[9]*x**8      
        #print(y)
        kernel = np.append(kernel, y)
    
    # if np.sum(kernel) != 0:
    #     kernel = kernel / np.sum(kernel)
    kernel = kernel / np.abs(kernel).max()
    
    plt.plot(kernel)
    plt.savefig(f"d:\\takahashi_k\\model\\analyze\\wall\\kernel\\spa=1\\radius({radius})_order({order}).png")
    plt.clf() 
    

if __name__ == '__main__':
    radius_list = [1, 2, 3, 4, 5, 7, 9, 11, 14, 17, 20, 25, 30]
    for r in radius_list:
        # _allmodel_kernel1d(radius=r, order=1, avespa=1)
        # _allmodel_kernel1d(radius=r, order=0, avespa=1)
        _nonwallmodel_kernel1d(radius=r, order=1, avespa=1)
        _nonwallmodel_kernel1d(radius=r, order=0, avespa=1)
        _wallmodel_kernel1d(radius=r, order=1, avespa=1)
        _wallmodel_kernel1d(radius=r, order=0, avespa=1)
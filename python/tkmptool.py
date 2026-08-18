# tkmptool.py
import numpy as np
import scipy as sc
import scipy.linalg as sclinalg
import mpmath
import rdd
import rcdd

def get_dtest_linear_eq(dimension, log10_max_d = 1.0, log10_min_d = -25.0, random_seed = 20210701):
    # 対角行列
    log10_max_d, log10_min_d = 1.0, -25.0
    log10_step_d = (log10_max_d - log10_min_d) / (dimension)
    log10_d = np.arange(log10_min_d, log10_max_d, log10_step_d)
    mat_true_eigenvalues = np.diag([10**i for i in log10_d])
    #print('D = \n', mat_d)

    # 乱数行列
    np.random.seed(random_seed)
    # mat_t = sc.random.rand(dimension, dimension) # Old
    mat_t = np.random.rand(dimension, dimension)
    mat_t_inv = sclinalg.inv(mat_t)
    #print('T = \n', mat_t)

    # テスト行列: A := T * D * T(-1)
    mat_a = mat_t @ mat_true_eigenvalues * mat_t_inv
    #print('A = \n', mat_a)

    # A * x = b
    vec_true_x = np.array([i + 1 for i in range(dimension)])
    vec_b = mat_a * vec_true_x

    # A, b, x, eig
    return mat_a, vec_b, vec_true_x, mat_true_eigenvalues

# mpmathnify dmat
def init_convert_mp(dmat):
    #row_dim, col_dim = dmat.shape
    #print(row_dim, col_dim)
    dim = len(dmat)
    #mp_dmat = mpmath.matrix(row_dim, col_dim)
    mp_dmat = [mpmath.mpmathify(dmat[i]) for i in range(dim)]
    #for i in range(dim):
    #    for j in range(dim):
    #        mp_dmat[i, j] = mpmath.mpmathify(dmat[i, j])
    # print('mp.D = \n', mp_mat_d)1
    return mp_dmat

# mpmathnify dmat
def init_convert_mpmatrix(dmat):
    row_dim, col_dim = dmat.shape
    #print(row_dim, col_dim)
    #dim = len(dmat)
    mp_dmat = mpmath.matrix(row_dim, col_dim)
    #mp_dmat = [mpmath.mpmathify(dmat[i]) for i in range(dim)]
    for i in range(row_dim):
        for j in range(col_dim):
            mp_dmat[i, j] = mpmath.mpmathify(dmat[i, j])
    # print('mp.D = \n', mp_mat_d)1
    return mp_dmat

# get max, min and norm relative errors
def relerr(approx_vec, true_vec):
    #reldiff = approx_vec - true_vec
    dim = len(approx_vec)
    if dim > len(true_vec): dim = len(true_vec)
    reldiff = [approx_vec[i] - true_vec[i] for i in range(dim)]
    # print(reldiff)
    relnorm = mpmath.norm(reldiff) / mpmath.norm(true_vec)
    #print(relnorm)
    #for i in range(reldiff.rows):
    #    for j in range(reldiff.cols):
    for i in range(dim):
        if true_vec[i] != 0:
            reldiff[i] = mpmath.fabs(reldiff[i] / true_vec[i])
        else:
            reldiff[i] = mpmath.fabs(reldiff[i])

    # print(reldiff)
    return max(reldiff), min(reldiff), relnorm


def get_mptest_linear_eq(dimension, log10_max_d = 1.0, log10_min_d = -25.0, random_seed = 20210701):
    print('mpmath.mp.prec, dps = ', mpmath.mp.prec, mpmath.mp.dps)
    # 対角行列
    #log10_max_d, log10_min_d = 1.0, -25.0
    log10_step_d = (log10_max_d - log10_min_d) / (dimension)
    log10_d = np.arange(log10_min_d, log10_max_d, log10_step_d)
    mat_true_eigenvalues = np.diag([10**i for i in log10_d])
    #print('D = \n', mat_d)

    # 乱数行列
    np.random.seed(random_seed)
    #mat_t = sc.random.rand(dimension, dimension)
    mat_t = np.random.rand(dimension, dimension)
    mat_t_inv = sclinalg.inv(mat_t)
    #print('T = \n', mat_t)

    mp_mat_true_eigenvalues = init_convert_mpmatrix(mat_true_eigenvalues)
    mp_mat_t = init_convert_mpmatrix(mat_t)
    mp_mat_t_inv = mpmath.inverse(mp_mat_t)
    #print('mp.T = \n', mp_mat_t)
    #print('mp.T^(-1) = \n', mp_mat_t_inv)
    mp_mat_a = mp_mat_t * mp_mat_true_eigenvalues * mp_mat_t_inv
    #print('mp.A = \n', mp_mat_a)

    # A * x = b
    #mp_vec_true_x = mpmath.matrix([mpmath.mpmathify(i) for i in range(dimension)])
    mp_vec_true_x = mpmath.matrix([mpmath.mpmathify(i + 1) for i in range(dimension)])
    mp_vec_b = mp_mat_a * mp_vec_true_x

    # A, b, x, eig
    return mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_true_eigenvalues

# write mpvector to file
def write_mpvector(filename, mpvector, delimiter = ', '):
    with open(filename, 'w') as fp:
        dim = len(mpvector)
        for i in range(dim):
            fp.write(str(i) + delimiter + mpmath.nstr(mpvector[i, 0], mpmath.mp.dps) + '\n')

    return

# https://stackoverflow.com/questions/21224810/python-alternative-to-fscanf-c-code
# read mpvector from file
def read_mpvector(filename, delimiter = ', '):
    dim = 0
    # know row_dim and col_dim
    with open(filename, 'r') as fp:
        for line in fp:
            index, val = line.split(delimiter)
            i = int(index)
            if i > dim: dim = i 

    dim += 1
    #print('dim = ', dim)
    ret_mpvec = mpmath.matrix(dim, 1)
    with open(filename, 'r') as fp:
        for line in fp:
            index, val = line.split(delimiter)
            i = int(index)
            ret_mpvec[i, 0] = mpmath.mpmathify(val)
            #print(i, ret_mpvec[i])
        
    return ret_mpvec

# write mpmatrix to file
def write_mpmatrix(filename, mpmatrix, delimiter = ', '):
    with open(filename, 'w') as fp:
        row_dim, col_dim = mpmatrix.rows, mpmatrix.cols
        for i in range(row_dim):
            for j in range(col_dim):
                fp.write(str(i) + delimiter + str(j) + delimiter + mpmath.nstr(mpmatrix[i, j], mpmath.mp.dps) + '\n')

    return

# https://stackoverflow.com/questions/21224810/python-alternative-to-fscanf-c-code
# read mpmatrix from file
def read_mpmatrix(filename, delimiter = ', '):
    row_dim, col_dim = 0, 0
    # know row_dim and col_dim
    with open(filename, 'r') as fp:
        for line in fp:
            row_index, col_index, val = line.split(delimiter)
            i = int(row_index)
            j = int(col_index)
            if i > row_dim: row_dim = i 
            if j > col_dim: col_dim = j

    row_dim += 1
    col_dim += 1
    #print('row_dim, col_dim = ', row_dim, col_dim)
    ret_mpmat = mpmath.matrix(row_dim, col_dim)
    with open(filename, 'r') as fp:
        for line in fp:
            row_index, col_index, val = line.split(delimiter)
            i = int(row_index)
            j = int(col_index)
            ret_mpmat[i, j] = mpmath.mpmathify(val)
            #print(i, j, ret_mpmat[i, j])
        
    return ret_mpmat


def get_mptest_linear_eq_complex(dimension, log10_max_d = 1.0, log10_min_d = -25.0, random_seed = 20210701):
    print('mpmath.mp.prec, dps = ', mpmath.mp.prec, mpmath.mp.dps)
    # 対角行列
    #log10_max_d, log10_min_d = 1.0, -25.0
    log10_step_d = (log10_max_d - log10_min_d) / (dimension)
    log10_d = np.arange(log10_min_d, log10_max_d, log10_step_d)
    mat_true_eigenvalues = np.diag([10**i for i in log10_d])
    #print('D = \n', mat_d)

    # 乱数行列
    np.random.seed(random_seed)
    #mat_t = sc.random.rand(dimension, dimension)
    mat_t_re = np.random.rand(dimension, dimension)
    mat_t_re_inv = sclinalg.inv(mat_t_re)
    mat_t_im = np.random.rand(dimension, dimension)
    mat_t_im_inv = sclinalg.inv(mat_t_im)
    #print('T = \n', mat_t_re, mat_t_im)

    mp_mat_true_eigenvalues = init_convert_mpmatrix(mat_true_eigenvalues)
    mp_mat_t_re = init_convert_mpmatrix(mat_t_re)
    mp_mat_t_re_inv = mpmath.inverse(mp_mat_t_re)
    mp_mat_t_im = init_convert_mpmatrix(mat_t_im)
    mp_mat_t_im_inv = mpmath.inverse(mp_mat_t_im)

    #print('mp.T = \n', mp_mat_t)
    #print('mp.T^(-1) = \n', mp_mat_t_inv)
    mp_mat_a_re = mp_mat_t_re * mp_mat_true_eigenvalues * mp_mat_t_re_inv
    mp_mat_a_im = mp_mat_t_im * mp_mat_true_eigenvalues * mp_mat_t_im_inv
    mp_mat_a = mp_mat_a_re + mp_mat_a_im * 1j
    #print('mp.A = \n', mp_mat_a)

    # A * x = b
    #mp_vec_true_x_re = mpmath.matrix([mpmath.mpmathify(i) for i in range(dimension)])
    #mp_vec_true_x_im = mpmath.matrix([mpmath.mpmathify(i) for i in range(dimension)])
    mp_vec_true_x_re = mpmath.matrix([mpmath.mpmathify(i + 1) for i in range(dimension)])
    mp_vec_true_x_im = mpmath.matrix([mpmath.mpmathify(i + 1) for i in range(dimension)])
    mp_vec_true_x = mp_vec_true_x_re + mp_vec_true_x_im * 1j

    mp_vec_b_re = mp_mat_a_re * mp_vec_true_x_re - mp_mat_a_im * mp_vec_true_x_im
    mp_vec_b_im = mp_mat_a_im * mp_vec_true_x_re + mp_mat_a_re * mp_vec_true_x_im
    mp_vec_b = mp_vec_b_re + mp_vec_b_im * 1j

    # A, b, x, eig
    # return mp_mat_a_re, mp_mat_a_im, mp_vec_b_re, mp_vec_b_im, mp_vec_true_x_re, mp_vec_true_x_im, mp_mat_true_eigenvalues
    return mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_true_eigenvalues


# mptest_matrix.py: Multiple Precision Test Matrix
from tkmptool import get_dtest_linear_eq, get_mptest_linear_eq
import numpy as np
import scipy as sc
import scipy.linalg as sclinalg
import mpmath

# tkmptool.py
from tkmptool import *

# 次元数
#dim = 3
#dim = 100
str_dim = input('Dimension = ?')
dim = int(str_dim)

# get double prec. linear equation for test
# def get_dtest_linear_eq(dimension, log10_max_d = 1.0, log10_min_d = -25.0, random_seed = 20210701):
#mat_a, b, true_x, mat_d = get_dtest_linear_eq(dim)
mat_a, b, true_x, mat_d = get_dtest_linear_eq(dim, log10_min_d = -20)

# 検算:固有値
eig, ev = sclinalg.eig(mat_a)
eig = np.real(eig) # 虚数部カット
print('eig(A) = \n', max(eig), min(eig))
ddiag = [mat_d[i,i] for i in range(dim)]
print('D      = \n', max(ddiag), min(ddiag))

#mpmath.mp.dps = 30 # 10進30桁
mpmath.mp.dps = 50 # 10進50桁

# get double prec. linear equation for test
# mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq(dim)
mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq(dim, log10_min_d = -20)

mp_eig, mp_ev = mpmath.eig(mp_mat_a)
#print('mp_eig = ', mp_eig)
mp_eig = [mp_eig[i].real for i in range(len(mp_eig))]
#print('mp.eig(A) = \n', max(mp_eig), min(mp_eig)) #mpmath.eig_sort(mp_eig))
mp_ddiag = [mp_mat_d[i, i] for i in range(dim)]
#print(type(mp_ddiag), type(mp_eig))
#print('mp.D      = \n', max(mp_ddiag), min(mp_ddiag))
max_rel, min_rel, norm_rel = relerr(sorted(mp_eig), sorted(mp_ddiag))
print('max, min, norm_relerr_eig: ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# LU 分解
mp_vec_x = mpmath.lu_solve(mp_mat_a, mp_vec_b)
#print('x = \n', mp_vec_x)
max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

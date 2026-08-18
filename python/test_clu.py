# test_clu.py: 複素直接法のPythonスクリプト

# BNCamPyを読み込み
from bncampy import *
# 自作ツールを読み込み
from tkmptool import *

# RDD.pyを初期化
librdd.rdd_start()

# 次元数をセット
input_dim = input('Input dim = ')
input_dim = int(input_dim)
row_dim = mid_dim = col_dim = input_dim

# ------
# DD
# ----- 
# mpmathの多倍長精度をセット
mpmath.mp.prec = 106 # DD = 53 * 2
print('mpmath.prec = ', mpmath.mp.prec, 'mpmath.dps = ', mpmath.mp.dps)

# 複素連立一次方程式をセット
mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq_complex(mid_dim)

# 多倍長精度クラスによる複素ベクトル・行列の宣言
mp_vec_x = [mpc_zero] * (mid_dim)
cddfloat_vec_true_x = [cdd_zero] * (mid_dim) 
cddfloat_vec_b = [cdd_zero] * (mid_dim)
cddfloat_mat_a = [cdd_zero] * (row_dim * mid_dim)

# BNCmatmulの複素DD精度ベクトル(CDDVector)を宣言
cddvec_true_x = init_cddvector(mid_dim)
cddvec_b = init_cddvector(row_dim)
cddvec_x = init_cddvector(mid_dim)

# mpmath.mpcクラスからcdd_floatクラスへの変換
for i in range(mid_dim):
	cddfloat_vec_true_x[i] = mpc_get_cdd_float(mp_vec_true_x[i])

# cdd_floatクラスからCDDVector型(BNCmatmul)への変換
set_cddvector_cdd_float(cddvec_true_x, cddfloat_vec_true_x)

# mpmath.mpcクラスからcdd_floatクラスへの変換 
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j
		cddfloat_mat_a[ij_index] = mpc_get_cdd_float(mp_mat_a[i,j])

# CDDMatrix(BNCmatmul)の宣言とcdd_float型からの変換
cddmat_a = init_cddmatrix(row_dim, mid_dim)
set_cddmatrix_cdd_float(cddmat_a, cddfloat_mat_a)

# 複素DD行列・ベクトル乗算(BNCmatmul)
start_time = time.time()
mul_cddmatrix_cddvec(cddvec_b, cddmat_a, cddvec_true_x)
end_time = time.time()
dd_mvmul_time = end_time - start_time

# LU 分解(mpmath)
start_time = time.time()
mp_vec_x = mpmath.lu_solve(mp_mat_a, mp_vec_b)
end_time = time.time()
mp_lusolve_time = end_time - start_time

# 直接法計算時間表示
print(f'mp_LU = {mp_lusolve_time:10.3g}')
# 最大，最小，ノルム相対誤差
max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
print('mpmath: max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# CDDLU分解と前進・後退代入(BNCmatmul)
ptr_pivot = (ct.c_long * (mid_dim))()
start_time = time.time()
ret_ddlu = CDDLUdecompPM(cddmat_a, ptr_pivot)
#ret_ddlu = CDDLUdecompP(cddmat_a, ptr_pivot) # 部分
SolveCDDLSPM(cddvec_x, cddmat_a, cddvec_b, ptr_pivot)
#SolveCDDLSP(cddvec_x, cddmat_a, cddvec_b, ptr_pivot)
end_time = time.time()
dd_lusolve_time = end_time - start_time

# 直接法計算時間表示
print(f'dd_LU = {dd_lusolve_time:10.3g}')

# CDDVector -> cdd_floatクラス -> mpmath.mpcへの変換
for i in range(mid_dim):
	mp_vec_x[i] = mpc_set_cdd_float(rcdd.cdd_float(
		cddvec_x.contents.re.contents.element[0][i],
		cddvec_x.contents.re.contents.element[1][i],
		cddvec_x.contents.im.contents.element[0][i],
		cddvec_x.contents.im.contents.element[1][i]
	))
	mp_vec_true_x[i] = mpc_set_cdd_float(rcdd.cdd_float(
		cddvec_true_x.contents.re.contents.element[0][i], 
		cddvec_true_x.contents.re.contents.element[1][i],
		cddvec_true_x.contents.im.contents.element[0][i], 
		cddvec_true_x.contents.im.contents.element[1][i]
	))

# 最大，最小，ノルム相対誤差
max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
print('CDD   : max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# 変数を削除
del cddfloat_vec_true_x, cddfloat_vec_b
free_cddvector(cddvec_x); free_cddvector(cddvec_b)
del mp_mat_a
del cddfloat_mat_a
free_cddmatrix(cddmat_a)

# RDD.pyの使用を終了
librdd.rdd_end()

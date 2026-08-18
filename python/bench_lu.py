# load BNCamPy
from bncampy import *
from tkmptool import *

# start
librdd.rdd_start()

input_dim = input('Input dim = ')
input_dim = int(input_dim)
row_dim = mid_dim = col_dim = input_dim

# mpmath_skip_flag
#mpmath_skip_flag = True
mpmath_skip_flag = False

# ------
# DD
# ----- 
# mpfr
mpmath.mp.prec = 106 # DD
mp_sqrt2 = mpmath.mp.sqrt(mpmath.mp.mpf(2))
mp_sqrt3 = mpmath.mp.sqrt(mpmath.mp.mpf(3))
mp_sqrt5 = mpmath.mp.sqrt(mpmath.mp.mpf(5))
print('mpmath.prec = ', mpmath.mp.prec, 'mpmath.dps = ', mpmath.mp.dps)

mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq(mid_dim)
mp_vec_x = [mp_zero] * (mid_dim)

# Vector variables
dd_vec_true_x = [dd_zero] * (mid_dim)
ptr_dd_vec_true_x = (ct.c_double * (mid_dim * 2))()
dd_vec_b = [dd_zero] * (mid_dim)
ptr_dd_vec_b = (ct.c_double * (row_dim * 2))()

# set elements of vectors
for i in range(mid_dim):
	#dd_vec_a[i] = rdd.mpfr_get_dd(mpfr_sqrt5 * (i + 1))
	dd_vec_true_x[i] = mpf_get_dd(mp_vec_true_x[i])
	ptr_dd_vec_true_x[i * 2]     = dd_vec_true_x[i].val[0]
	ptr_dd_vec_true_x[i * 2 + 1] = dd_vec_true_x[i].val[1]

# ddvector
ddvec_true_x = init_ddvector(mid_dim)
set_ddvector_ddfloat(ddvec_true_x, ptr_dd_vec_true_x, mid_dim)
#for i in range(mid_dim):
#	print('true_x[', i, '] = ', rdd.dd_float(ddvec_true_x.contents.element[0][i], ddvec_true_x.contents.element[1][i]))

ddvec_b = init_ddvector(row_dim)
ddvec_x = init_ddvector(mid_dim)

# Matrix variables
dd_mat_a = [dd_zero] * (row_dim * mid_dim)
ptr_dd_mat_a = (ct.c_double * (row_dim * mid_dim * 2))()

# set matrices
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j

		# set elements of A
		# A := sqrt(2) * [(i + j + 1)]
		#dd_mat_a[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt2 * (i + j + 1))
		dd_mat_a[ij_index] = mpf_get_dd(mp_mat_a[i,j])
		ptr_dd_mat_a[(ij_index) * 2]     = dd_mat_a[ij_index].val[0]
		ptr_dd_mat_a[(ij_index) * 2 + 1] = dd_mat_a[ij_index].val[1]


# ddmatrix
ddmat_a = init_ddmatrix(row_dim, mid_dim)
set_ddmatrix_ddfloat(ddmat_a, ptr_dd_mat_a, row_dim * mid_dim)

# mat_ddmatrix_ddvec
start_time = time.time()
mul_ddmatrix_ddvec(ddvec_b, ddmat_a, ddvec_true_x)
end_time = time.time()
dd_mvmul_time = end_time - start_time

# LU 分解
if mpmath_skip_flag != True:
	start_time = time.time()
	mp_vec_x = mpmath.lu_solve(mp_mat_a, mp_vec_b)
	end_time = time.time()
	mp_lusolve_time = end_time - start_time
	#print('x = \n', mp_vec_x)
	max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
	print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

#LU
ptr_pivot = (ct.c_long * (mid_dim))()
start_time = time.time()
ret_ddlu = DDLUdecompPM(ddmat_a, ptr_pivot)
#ret_ddlu = DDLUdecompP(ddmat_a, ptr_pivot)
SolveDDLSPM(ddvec_x, ddmat_a, ddvec_b, ptr_pivot)
#SolveDDLSP(ddvec_x, ddmat_a, ddvec_b, ptr_pivot)
end_time = time.time()
dd_lusolve_time = end_time - start_time
#print('ret of LU = ', ret_ddlu)
print('dd_LU = ', dd_lusolve_time)
if mpmath_skip_flag != True: print('mp_LU = ', mp_lusolve_time)

#for i in range(mid_dim):
#	print('x[', i, '] = ', ddvec_x.contents.element[0][i])
for i in range(mid_dim):
	mp_vec_x[i] = mpf_set_dd(rdd.dd_float(ddvec_x.contents.element[0][i], ddvec_x.contents.element[1][i]))
	mp_vec_true_x[i] = mpf_set_dd(rdd.dd_float(ddvec_true_x.contents.element[0][i], ddvec_true_x.contents.element[1][i]))

max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# delete
del dd_vec_true_x, dd_vec_b, ptr_dd_vec_true_x, ptr_dd_vec_b
free_ddvector(ddvec_x); free_ddvector(ddvec_b);
del dd_mat_a
del mp_mat_a
del ptr_dd_mat_a
free_ddmatrix(ddmat_a)

# ------
# TD
# ----- 
# mpfr
mpmath.mp.prec = 159 # TD
mp_sqrt2 = mpmath.mp.sqrt(mpmath.mp.mpf(2))
mp_sqrt3 = mpmath.mp.sqrt(mpmath.mp.mpf(3))
mp_sqrt5 = mpmath.mp.sqrt(mpmath.mp.mpf(5))
print('mpmath.prec = ', mpmath.mp.prec, 'mpmath.dps = ', mpmath.mp.dps)

mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq(mid_dim)

# Vector variables
td_vec_true_x = [td_zero] * (mid_dim)
ptr_td_vec_true_x = (ct.c_double * (mid_dim * 3))()
td_vec_b = [td_zero] * (mid_dim)
ptr_td_vec_b = (ct.c_double * (row_dim * 3))()

# set elements of vectors
for i in range(mid_dim):
	#dd_vec_a[i] = rdd.mpfr_get_dd(mpfr_sqrt5 * (i + 1))
	td_vec_true_x[i] = mpf_get_td(mp_vec_true_x[i])
	ptr_td_vec_true_x[i * 3]     = td_vec_true_x[i].val[0]
	ptr_td_vec_true_x[i * 3 + 1] = td_vec_true_x[i].val[1]
	ptr_td_vec_true_x[i * 3 + 2] = td_vec_true_x[i].val[2]

# tdvector
tdvec_true_x = init_tdvector(mid_dim)
set_tdvector_tdfloat(tdvec_true_x, ptr_td_vec_true_x, mid_dim)
#for i in range(mid_dim):
#	print('true_x[', i, '] = ', rdd.dd_float(ddvec_true_x.contents.element[0][i], ddvec_true_x.contents.element[1][i]))

tdvec_b = init_tdvector(row_dim)
tdvec_x = init_tdvector(mid_dim)

# Matrix variables
td_mat_a = [td_zero] * (row_dim * mid_dim)
ptr_td_mat_a = (ct.c_double * (row_dim * mid_dim * 3))()

# set matrices
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j

		# set elements of A
		# A := sqrt(2) * [(i + j + 1)]
		#dd_mat_a[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt2 * (i + j + 1))
		td_mat_a[ij_index] = mpf_get_td(mp_mat_a[i,j])
		ptr_td_mat_a[(ij_index) * 3]     = td_mat_a[ij_index].val[0]
		ptr_td_mat_a[(ij_index) * 3 + 1] = td_mat_a[ij_index].val[1]
		ptr_td_mat_a[(ij_index) * 3 + 2] = td_mat_a[ij_index].val[2]


# tdmatrix
tdmat_a = init_tdmatrix(row_dim, mid_dim)
set_tdmatrix_tdfloat(tdmat_a, ptr_td_mat_a, row_dim * mid_dim)

# mat_ddmatrix_ddvec
start_time = time.time()
mul_tdmatrix_tdvec(tdvec_b, tdmat_a, tdvec_true_x)
end_time = time.time()
td_mvmul_time = end_time - start_time

# LU 分解
if mpmath_skip_flag != True:
	start_time = time.time()
	mp_vec_x = mpmath.lu_solve(mp_mat_a, mp_vec_b)
	end_time = time.time()
	mp_lusolve_time = end_time - start_time
	#print('x = \n', mp_vec_x)
	max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
	print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

#LU
ptr_pivot = (ct.c_long * (mid_dim))()
start_time = time.time()
ret_tdlu = TDLUdecompPM(tdmat_a, ptr_pivot)
#ret_tdlu = TDLUdecompP(tdmat_a, ptr_pivot)
SolveTDLSPM(tdvec_x, tdmat_a, tdvec_b, ptr_pivot)
#SolveTDLSP(tdvec_x, tdmat_a, tdvec_b, ptr_pivot)
end_time = time.time()
td_lusolve_time = end_time - start_time
#print('ret of LU = ', ret_ddlu)
#print('pivot = ', [ptr_pivot[i] for i in range(mid_dim)])
print('td_LU = ', td_lusolve_time)
if mpmath_skip_flag != True: print('mp_LU = ', mp_lusolve_time)

#for i in range(mid_dim):
#	print('x[', i, '] = ', ddvec_x.contents.element[0][i])
for i in range(mid_dim):
	mp_vec_x[i] = mpf_set_td(rdd.td_float(tdvec_x.contents.element[0][i], tdvec_x.contents.element[1][i], tdvec_x.contents.element[2][i]))
	mp_vec_true_x[i] = mpf_set_td(rdd.td_float(tdvec_true_x.contents.element[0][i], tdvec_true_x.contents.element[1][i], tdvec_true_x.contents.element[2][i]))

max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# delete
del td_vec_true_x, td_vec_b, ptr_td_vec_true_x, ptr_td_vec_b
free_tdvector(tdvec_x); free_tdvector(tdvec_b);
del td_mat_a
del mp_mat_a
del ptr_td_mat_a
free_tdmatrix(tdmat_a)

# ------
# QD
# ----- 
# mpfr
mpmath.mp.prec = 212 # QD
mp_sqrt2 = mpmath.mp.sqrt(mpmath.mp.mpf(2))
mp_sqrt3 = mpmath.mp.sqrt(mpmath.mp.mpf(3))
mp_sqrt5 = mpmath.mp.sqrt(mpmath.mp.mpf(5))
print('mpmath.prec = ', mpmath.mp.prec, 'mpmath.dps = ', mpmath.mp.dps)

mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq(mid_dim)

# Vector variables
qd_vec_true_x = [qd_zero] * (mid_dim)
ptr_qd_vec_true_x = (ct.c_double * (mid_dim * 4))()
qd_vec_b = [qd_zero] * (mid_dim)
ptr_qd_vec_b = (ct.c_double * (row_dim * 4))()

# set elements of vectors
for i in range(mid_dim):
	#dd_vec_a[i] = rdd.mpfr_get_dd(mpfr_sqrt5 * (i + 1))
	qd_vec_true_x[i] = mpf_get_qd(mp_vec_true_x[i])
	ptr_qd_vec_true_x[i * 4]     = qd_vec_true_x[i].val[0]
	ptr_qd_vec_true_x[i * 4 + 1] = qd_vec_true_x[i].val[1]
	ptr_qd_vec_true_x[i * 4 + 2] = qd_vec_true_x[i].val[2]
	ptr_qd_vec_true_x[i * 4 + 3] = qd_vec_true_x[i].val[3]

# qdvector
qdvec_true_x = init_qdvector(mid_dim)
set_qdvector_qdfloat(qdvec_true_x, ptr_qd_vec_true_x, mid_dim)
#for i in range(mid_dim):
#	print('true_x[', i, '] = ', rdd.dd_float(ddvec_true_x.contents.element[0][i], ddvec_true_x.contents.element[1][i]))

qdvec_b = init_qdvector(row_dim)
qdvec_x = init_qdvector(mid_dim)

# Matrix variables
qd_mat_a = [qd_zero] * (row_dim * mid_dim)
ptr_qd_mat_a = (ct.c_double * (row_dim * mid_dim * 4))()

# set matrices
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j

		# set elements of A
		# A := sqrt(2) * [(i + j + 1)]
		#dd_mat_a[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt2 * (i + j + 1))
		qd_mat_a[ij_index] = mpf_get_qd(mp_mat_a[i,j])
		ptr_qd_mat_a[(ij_index) * 4]     = qd_mat_a[ij_index].val[0]
		ptr_qd_mat_a[(ij_index) * 4 + 1] = qd_mat_a[ij_index].val[1]
		ptr_qd_mat_a[(ij_index) * 4 + 2] = qd_mat_a[ij_index].val[2]
		ptr_qd_mat_a[(ij_index) * 4 + 3] = qd_mat_a[ij_index].val[3]

# qdmatrix
qdmat_a = init_qdmatrix(row_dim, mid_dim)
set_qdmatrix_qdfloat(qdmat_a, ptr_qd_mat_a, row_dim * mid_dim)

# mat_ddmatrix_ddvec
start_time = time.time()
mul_qdmatrix_qdvec(qdvec_b, qdmat_a, qdvec_true_x)
end_time = time.time()
qd_mvmul_time = end_time - start_time

# LU 分解
if mpmath_skip_flag != True:
	start_time = time.time()
	mp_vec_x = mpmath.lu_solve(mp_mat_a, mp_vec_b)
	end_time = time.time()
	mp_lusolve_time = end_time - start_time
	#print('x = \n', mp_vec_x)
	max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
	print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

#LU
ptr_pivot = (ct.c_long * (mid_dim))()
start_time = time.time()
ret_qdlu = QDLUdecompPM(qdmat_a, ptr_pivot)
#ret_qdlu = QDLUdecompP(qdmat_a, ptr_pivot)
SolveQDLSPM(qdvec_x, qdmat_a, qdvec_b, ptr_pivot)
#SolveQDLSP(qdvec_x, qdmat_a, qdvec_b, ptr_pivot)
end_time = time.time()
qd_lusolve_time = end_time - start_time
#print('ret of LU = ', ret_ddlu)
#print('pivot = ', [ptr_pivot[i] for i in range(mid_dim)])
print('qd_LU = ', qd_lusolve_time)
if mpmath_skip_flag != True: print('mp_LU = ', mp_lusolve_time)

#for i in range(mid_dim):
#	print('x[', i, '] = ', ddvec_x.contents.element[0][i])
for i in range(mid_dim):
	mp_vec_x[i] = mpf_set_qd(rdd.qd_float(qdvec_x.contents.element[0][i], qdvec_x.contents.element[1][i], qdvec_x.contents.element[2][i], qdvec_x.contents.element[3][i]))
	mp_vec_true_x[i] = mpf_set_qd(rdd.qd_float(qdvec_true_x.contents.element[0][i], qdvec_true_x.contents.element[1][i], qdvec_true_x.contents.element[2][i], qdvec_true_x.contents.element[3][i]))

max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# delete
del qd_vec_true_x, qd_vec_b, ptr_qd_vec_true_x, ptr_qd_vec_b
free_qdvector(qdvec_x); free_qdvector(qdvec_b);
del qd_mat_a
del mp_mat_a
del ptr_qd_mat_a
free_qdmatrix(qdmat_a)

librdd.rdd_end();

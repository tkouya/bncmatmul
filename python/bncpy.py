# bench_matmul.py: LIBRDD test

import ctypes as ct
import rdd
import math
import gmpy2

import numpy as np
import mpmath
import time

# Profile
import cProfile

# mpmath backend
print('mpmath.libmp.BACKEND = ', mpmath.libmp.BACKEND)
mpmath.mp.prec = 212
mp_sqrt2 = mpmath.mp.sqrt(mpmath.mp.mpf(2))
mp_sqrt3 = mpmath.mp.sqrt(mpmath.mp.mpf(3))
mp_sqrt5 = mpmath.mp.sqrt(mpmath.mp.mpf(5))
print('mpmath.prec = ', mpmath.mp.prec, 'mpmath.dps = ', mpmath.mp.dps)

# gmpy2
#print(gmpy2.get_context())
#gmpy2.get_context().precision = 1024 # in bits
gmpy2.get_context().precision = 212 # in bits
#gmpy2.get_context().precision = 106 # in bits
print(gmpy2.get_context())
#print(gmpy2.sqrt(gmpy2.mpfr(2)))

mpfr_sqrt2 = gmpy2.sqrt(gmpy2.mpfr(2))
mpfr_sqrt3 = gmpy2.sqrt(gmpy2.mpfr(3))
mpfr_sqrt5 = gmpy2.sqrt(gmpy2.mpfr(5))

#print('a = ', a)
print('mpfr_sqrt2 = ', mpfr_sqrt2)
print('mpfr_sqrt3 = ', mpfr_sqrt3)
print('mpfr_sqrt5 = ', mpfr_sqrt5)


# RDD library
librdd = ct.cdll.LoadLibrary('librdd.so');

# BNCmatmul
#libbncmatmul = ct.cdll.LoadLibrary('libbncmatmul-0.21.so')
libbncmatmul = ct.cdll.LoadLibrary('libbncmatmul-0.21_avx2.so')

print(libbncmatmul)

# https://stackoverflow.com/questions/25480492/how-to-operate-c-type-pointer-return-by-c-function-in-python
# DDVector class
# DD vector
#typedef struct {
#	long int dim; // dim <= real_dim
#	long int real_dim; // multiplier of _BNC_D_WIDTH
#    double *element[DDSIZE];
#} ddvector;
#typedef ddvector *DDVector;
class ddvector(ct.Structure):
	_fields_ = [
		("dim", ct.c_long),
		("real_dim", ct.c_long),
		("element", ct.ARRAY(ct.POINTER(ct.c_double), 2))
	]

# init_ddvector
init_ddvector = libbncmatmul.init_ddvector
init_ddvector.argtype = [ct.c_long]
init_ddvector.restype = ct.POINTER(ddvector)

# free_ddvector
free_ddvector = libbncmatmul.free_ddvector
free_ddvector.argtype = ct.POINTER(ddvector)
free_ddvector.restype = None

# dd array to ddvector
# DDVector vec -> ddfloat array
# void set_ddfloat_ddvec(ddfloat ret[], int ret_dim, DDVector vec);
set_ddfloat_ddvec = libbncmatmul.set_ddfloat_ddvec
set_ddfloat_ddvec.argtype = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ddvector)]
set_ddfloat_ddvec.restype = None

# ddfloat array -> DDVector ret
# void set_ddvector_ddfloat(DDVector ret, ddfloat array[], int array_dim);
set_ddvector_ddfloat = libbncmatmul.set_ddvector_ddfloat
set_ddvector_ddfloat.argtype = [ct.POINTER(ddvector), ct.POINTER(ct.c_double), ct.c_int]
set_ddvector_ddfloat.restype = None

# DDMatrix class
# DD matrix
#typedef struct{
#	long int row_dim, col_dim;
#	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
#	double *element[DDSIZE];
#} ddmatrix;
class ddmatrix(ct.Structure):
	_fields_ = [
		("row_dim", ct.c_long),
		("col_dim", ct.c_long),
		("real_row_dim", ct.c_long),
		("real_col_dim", ct.c_long),
		("element", ct.ARRAY(ct.POINTER(ct.c_double), 2))
	]

# init_ddmatrix
init_ddmatrix = libbncmatmul.init_ddmatrix
init_ddmatrix.argtype = [ct.c_long, ct.c_long]
init_ddmatrix.restype = ct.POINTER(ddmatrix)

# free_ddmatrix
free_ddmatrix = libbncmatmul.free_ddmatrix
free_ddmatrix.argtype = ct.POINTER(ddmatrix)
free_ddmatrix.restype = None

# dd array to ddvector
# DDMatrix mat -> ddfloat array
# void set_ddfloat_ddmat(ddfloat ret[], int ret_dim, DDMatrix mat);
set_ddfloat_ddmat = libbncmatmul.set_ddfloat_ddmat
set_ddfloat_ddmat.argtype = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ddmatrix)]
set_ddfloat_ddmat.restype = None

# ddfloat array -> DDmatrix ret
# void set_ddmatrix_ddfloat(DDMatrix ret, ddfloat array[], int array_dim);
set_ddmatrix_ddfloat = libbncmatmul.set_ddmatrix_ddfloat
set_ddmatrix_ddfloat.argtype = [ct.POINTER(ddmatrix), ct.POINTER(ct.c_double), ct.c_int]
set_ddmatrix_ddfloat.restype = None

# Matrix-vector multiplicaiton
#void mul_ddmatrix_ddvec(DDVector ret, DDMatrix mat, DDVector vec);
mul_ddmatrix_ddvec = libbncmatmul.mul_ddmatrix_ddvec
mul_ddmatrix_ddvec.argtype = [ct.POINTER(ddvector), ct.POINTER(ddmatrix), ct.POINTER(ddvector)]
mul_ddmatrix_ddvec.restype = None

# Block matrix multiplicaiton
#void mul_ddmatrix_block(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);
mul_ddmatrix_block = libbncmatmul.mul_ddmatrix_block
mul_ddmatrix_block.argtype = [ct.POINTER(ddmatrix), ct.POINTER(ddmatrix), ct.POINTER(ddmatrix), ct.c_long]
mul_ddmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_ddmatrix_strassen(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);
mul_ddmatrix_strassen = libbncmatmul.mul_ddmatrix_strassen
mul_ddmatrix_strassen.argtype = [ct.POINTER(ddmatrix), ct.POINTER(ddmatrix), ct.POINTER(ddmatrix), ct.c_long]
mul_ddmatrix_strassen.restype = None

# TDVector class
# TD vector
#typedef struct {
#	long int dim; // dim <= real_dim
#	long int real_dim; // multiplier of _BNC_D_WIDTH
#   double *element[TDSIZE];
#} tdvector;
#typedef tdvector *TDVector;
class tdvector(ct.Structure):
	_fields_ = [
		("dim", ct.c_long),
		("real_dim", ct.c_long),
		("element", ct.ARRAY(ct.POINTER(ct.c_double), 3))
	]

# init_tdvector
init_tdvector = libbncmatmul.init_tdvector
init_tdvector.argtype = [ct.c_long]
init_tdvector.restype = ct.POINTER(tdvector)

# free_tdvector
free_tdvector = libbncmatmul.free_tdvector
free_tdvector.argtype = ct.POINTER(tdvector)
free_tdvector.restype = None

# dd array to tdvector
# TDVector vec -> tdfloat array
# void set_tdfloat_tdvec(tdfloat ret[], int ret_dim, TDVector vec);
set_tdfloat_tdvec = libbncmatmul.set_tdfloat_tdvec
set_tdfloat_tdvec.argtype = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(tdvector)]
set_tdfloat_tdvec.restype = None

# tdfloat array -> TDVector ret
# void set_tdvector_tdfloat(TDVector ret, tdfloat array[], int array_dim);
set_tdvector_tdfloat = libbncmatmul.set_tdvector_tdfloat
set_tdvector_tdfloat.argtype = [ct.POINTER(tdvector), ct.POINTER(ct.c_double), ct.c_int]
set_tdvector_tdfloat.restype = None

# TDMatrix class
# TD matrix
#typedef struct{
#	long int row_dim, col_dim;
#	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
#	double *element[TDSIZE];
#} tdmatrix;
class tdmatrix(ct.Structure):
	_fields_ = [
		("row_dim", ct.c_long),
		("col_dim", ct.c_long),
		("real_row_dim", ct.c_long),
		("real_col_dim", ct.c_long),
		("element", ct.ARRAY(ct.POINTER(ct.c_double), 3))
	]

# init_tdmatrix
init_tdmatrix = libbncmatmul.init_tdmatrix
init_tdmatrix.argtype = [ct.c_long, ct.c_long]
init_tdmatrix.restype = ct.POINTER(tdmatrix)

# free_tdmatrix
free_tdmatrix = libbncmatmul.free_tdmatrix
free_tdmatrix.argtype = ct.POINTER(tdmatrix)
free_tdmatrix.restype = None

# td array to tdvector
# TDMatrix mat -> tdfloat array
# void set_tdfloat_tdmat(tdfloat ret[], int ret_dim, TDMatrix mat);
set_tdfloat_tdmat = libbncmatmul.set_tdfloat_tdmat
set_tdfloat_tdmat.argtype = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(tdmatrix)]
set_tdfloat_tdmat.restype = None

# tdfloat array -> TDmatrix ret
# void set_tdmatrix_tdfloat(DDMatrix ret, tdfloat array[], int array_dim);
set_tdmatrix_tdfloat = libbncmatmul.set_tdmatrix_tdfloat
set_tdmatrix_tdfloat.argtype = [ct.POINTER(tdmatrix), ct.POINTER(ct.c_double), ct.c_int]
set_tdmatrix_tdfloat.restype = None

# Matrix-vector multiplicaiton
#void mul_tdmatrix_tdvec(TDVector ret, TDMatrix mat, TDVector vec);
mul_tdmatrix_tdvec = libbncmatmul.mul_tdmatrix_tdvec
mul_tdmatrix_tdvec.argtype = [ct.POINTER(tdvector), ct.POINTER(tdmatrix), ct.POINTER(tdvector)]
mul_tdmatrix_tdvec.restype = None

# Block matrix multiplicaiton
#void mul_tdmatrix_block(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);
mul_tdmatrix_block = libbncmatmul.mul_tdmatrix_block
mul_tdmatrix_block.argtype = [ct.POINTER(tdmatrix), ct.POINTER(tdmatrix), ct.POINTER(tdmatrix), ct.c_long]
mul_tdmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_tdmatrix_strassen(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);
mul_tdmatrix_strassen = libbncmatmul.mul_tdmatrix_strassen
mul_tdmatrix_strassen.argtype = [ct.POINTER(tdmatrix), ct.POINTER(tdmatrix), ct.POINTER(tdmatrix), ct.c_long]
mul_tdmatrix_strassen.restype = None

# QDVector class
# QD vector
#typedef struct {
#	long int dim; // dim <= real_dim
#	long int real_dim; // multiplier of _BNC_D_WIDTH
#   double *element[QDSIZE];
#} qdvector;
#typedef qdvector *QDVector;
class qdvector(ct.Structure):
	_fields_ = [
		("dim", ct.c_long),
		("real_dim", ct.c_long),
		("element", ct.ARRAY(ct.POINTER(ct.c_double), 4))
	]

# init_qdvector
init_qdvector = libbncmatmul.init_qdvector
init_qdvector.argtype = [ct.c_long]
init_qdvector.restype = ct.POINTER(qdvector)

# free_qdvector
free_qdvector = libbncmatmul.free_qdvector
free_qdvector.argtype = ct.POINTER(qdvector)
free_qdvector.restype = None

# qd array to qdvector
# QDVector vec -> qdfloat array
# void set_qdfloat_qdvec(qdfloat ret[], int ret_dim, QDVector vec);
set_qdfloat_qdvec = libbncmatmul.set_qdfloat_qdvec
set_qdfloat_qdvec.argtype = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(qdvector)]
set_qdfloat_qdvec.restype = None

# qdfloat array -> QDVector ret
# void set_qdvector_qdfloat(QDVector ret, qdfloat array[], int array_dim);
set_qdvector_qdfloat = libbncmatmul.set_qdvector_qdfloat
set_qdvector_qdfloat.argtype = [ct.POINTER(qdvector), ct.POINTER(ct.c_double), ct.c_int]
set_qdvector_qdfloat.restype = None

# QDMatrix class
# QD matrix
#typedef struct{
#	long int row_dim, col_dim;
#	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
#	double *element[QDSIZE];
#} qdmatrix;
class qdmatrix(ct.Structure):
	_fields_ = [
		("row_dim", ct.c_long),
		("col_dim", ct.c_long),
		("real_row_dim", ct.c_long),
		("real_col_dim", ct.c_long),
		("element", ct.ARRAY(ct.POINTER(ct.c_double), 4))
	]

# init_qdmatrix
init_qdmatrix = libbncmatmul.init_qdmatrix
init_qdmatrix.argtype = [ct.c_long, ct.c_long]
init_qdmatrix.restype = ct.POINTER(qdmatrix)

# free_qdmatrix
free_qdmatrix = libbncmatmul.free_qdmatrix
free_qdmatrix.argtype = ct.POINTER(qdmatrix)
free_qdmatrix.restype = None

# qd array to qdvector
# QDMatrix mat -> qdfloat array
# void set_qdfloat_qdmat(qdfloat ret[], int ret_dim, QDMatrix mat);
set_qdfloat_qdmat = libbncmatmul.set_qdfloat_qdmat
set_qdfloat_qdmat.argtype = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(qdmatrix)]
set_qdfloat_qdmat.restype = None

# qdfloat array -> QDmatrix ret
# void set_qdmatrix_qdfloat(QDMatrix ret, qdfloat array[], int array_dim);
set_qdmatrix_qdfloat = libbncmatmul.set_qdmatrix_qdfloat
set_qdmatrix_qdfloat.argtype = [ct.POINTER(qdmatrix), ct.POINTER(ct.c_double), ct.c_int]
set_qdmatrix_qdfloat.restype = None

# Matrix-vector multiplicaiton
#void mul_qdmatrix_qdvec(QDVector ret, QDMatrix mat, QDVector vec);
mul_qdmatrix_qdvec = libbncmatmul.mul_qdmatrix_qdvec
mul_qdmatrix_qdvec.argtype = [ct.POINTER(qdvector), ct.POINTER(qdmatrix), ct.POINTER(qdvector)]
mul_qdmatrix_qdvec.restype = None

# Block matrix multiplicaiton
#void mul_qdmatrix_block(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);
mul_qdmatrix_block = libbncmatmul.mul_qdmatrix_block
mul_qdmatrix_block.argtype = [ct.POINTER(qdmatrix), ct.POINTER(qdmatrix), ct.POINTER(qdmatrix), ct.c_long]
mul_qdmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_qdmatrix_strassen(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);
mul_qdmatrix_strassen = libbncmatmul.mul_qdmatrix_strassen
mul_qdmatrix_strassen.argtype = [ct.POINTER(qdmatrix), ct.POINTER(qdmatrix), ct.POINTER(qdmatrix), ct.c_long]
mul_qdmatrix_strassen.restype = None

# start
librdd.rdd_start();

# 自作行列・ベクトル乗算
def xd_mymv(mat_a, row_dim_mat_a, col_dim_mat_a, vec_b, dim_vec_b, xd_zero):
	row_dim, col_dim = row_dim_mat_a, col_dim_mat_a

	zero = xd_zero

	if col_dim != dim_vec_b:
		print('A\'s col_dim = ', col_dim, ', b\'s dim = ', dim_vec_b, ' are mismatched!.')
		return [zero]

	vec_c = [zero] * row_dim

	for i in range(0, row_dim):
		vec_c[i] = zero
		for j in range(0, col_dim):
			vec_c[i] += mat_a[i * col_dim + j] * vec_b[j]

	return vec_c

# 自作行列乗算
def xd_mymatmul(mat_a, row_dim_mat_a, col_dim_mat_a, mat_b, row_dim_mat_b, col_dim_mat_b, xd_zero):
	row_dim  , mid_dim = row_dim_mat_a, col_dim_mat_a
	mid_dim_b, col_dim = row_dim_mat_b, col_dim_mat_b
 #   print('row_dim, mid_dim, col_dim = ', row_dim, mid_dim, col_dim)

	zero = xd_zero

	if mid_dim != mid_dim_b:
		print('A\'s col_dim = ', mid_dim, ', B\'s row_dim = ', mid_dim_b, ' are mismatched!.')
		return [zero]

	mat_c = [zero] * (row_dim * col_dim)

	for i in range(0, row_dim):
		for j in range(0, col_dim):
			ij_index = i * col_dim + j
			mat_c[ij_index] = zero
			for k in range(0, mid_dim):
				ik_index = i * mid_dim + k
				kj_index = k * mid_dim + j
				mat_c[ij_index] += mat_a[ik_index] * mat_b[kj_index]

	return mat_c


#for sq_dim in [32, 64, 128, 256]:
for sq_dim in [128, 256, 512, 1024]:

	# float

	# dimension
	row_dim = sq_dim
	mid_dim = sq_dim
	col_dim = sq_dim
	d_zero  = 0.0
	dd_zero = rdd.dd_float(0.0, 0.0)
	td_zero = rdd.td_float(0.0, 0.0, 0.0)
	qd_zero = rdd.qd_float(0.0, 0.0, 0.0, 0.0)
	mp_zero = mpmath.mp.mpf(0)
	mpfr_zero = gmpy2.mpfr(0)

	# ベクトルデータ型
	d_vec_a  = [ d_zero] * (mid_dim)
	dd_vec_a = [dd_zero] * (mid_dim)
	td_vec_a = [td_zero] * (mid_dim)
	qd_vec_a = [qd_zero] * (mid_dim)
	mp_vec_a = [mp_zero] * (mid_dim)
	mpfr_vec_a = [mpfr_zero] * (mid_dim)

	ptr_dd_vec_a = (ct.c_double * (mid_dim * 2))()
	ptr_td_vec_a = (ct.c_double * (mid_dim * 3))()
	ptr_qd_vec_a = (ct.c_double * (mid_dim * 4))()

	ptr_dd_vec_b = (ct.c_double * (row_dim * 2))()
	ptr_td_vec_b = (ct.c_double * (row_dim * 3))()
	ptr_qd_vec_b = (ct.c_double * (row_dim * 4))()

	for i in range(mid_dim):
		d_vec_a [i] = float(mpfr_sqrt5 * (i + 1))
		dd_vec_a[i] = rdd.mpfr_get_dd(mpfr_sqrt5 * (i + 1))
		td_vec_a[i] = rdd.mpfr_get_td(mpfr_sqrt5 * (i + 1))
		qd_vec_a[i] = rdd.mpfr_get_qd(mpfr_sqrt5 * (i + 1))
		mp_vec_a[i] = mp_sqrt5 * mpmath.mp.mpf(i + 1)
		mpfr_vec_a[i] = mpfr_sqrt5 * gmpy2.mpfr(i + 1)

		ptr_dd_vec_a[i * 2]     = dd_vec_a[i].val[0]
		ptr_dd_vec_a[i * 2 + 1] = dd_vec_a[i].val[1]
		ptr_td_vec_a[i * 3]     = td_vec_a[i].val[0]
		ptr_td_vec_a[i * 3 + 1] = td_vec_a[i].val[1]
		ptr_td_vec_a[i * 3 + 2] = td_vec_a[i].val[2]
		ptr_qd_vec_a[i * 4]     = qd_vec_a[i].val[0]
		ptr_qd_vec_a[i * 4 + 1] = qd_vec_a[i].val[1]
		ptr_qd_vec_a[i * 4 + 2] = qd_vec_a[i].val[2]
		ptr_qd_vec_a[i * 4 + 3] = qd_vec_a[i].val[3]


	# DD行列データ型
	#ptr_dd_mat = ct.c_double * (row_dim * mid_dim * 2)

	# A := sqrt(2) * [(i + j + 1)]
	d_mat_a  = [ d_zero] * (row_dim * mid_dim)
	dd_mat_a = [dd_zero] * (row_dim * mid_dim)
	td_mat_a = [td_zero] * (row_dim * mid_dim)
	qd_mat_a = [qd_zero] * (row_dim * mid_dim)
	mp_mat_a = [mp_zero] * (row_dim * mid_dim)
	mpfr_mat_a = [mpfr_zero] * (row_dim * mid_dim)

	ptr_dd_mat_a = (ct.c_double * (row_dim * mid_dim * 2))()
	ptr_td_mat_a = (ct.c_double * (row_dim * mid_dim * 3))()
	ptr_qd_mat_a = (ct.c_double * (row_dim * mid_dim * 4))()

#	print('size_mat_a = ', len(dd_mat_a))
	for i in range(row_dim):
		for j in range(mid_dim):
			ij_index = i * mid_dim + j
			d_mat_a [ij_index] = float(mpfr_sqrt2 * (i + j + 1))
			dd_mat_a[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt2 * (i + j + 1))
			td_mat_a[ij_index] = rdd.mpfr_get_td(mpfr_sqrt2 * (i + j + 1))
			qd_mat_a[ij_index] = rdd.mpfr_get_qd(mpfr_sqrt2 * (i + j + 1))
			mp_mat_a[ij_index] = mp_sqrt2 * mpmath.mp.mpf(i + j + 1)
			mpfr_mat_a[ij_index] = mpfr_sqrt2 * gmpy2.mpfr(i + j + 1)

			ptr_dd_mat_a[(ij_index) * 2]     = dd_mat_a[ij_index].val[0]
			ptr_dd_mat_a[(ij_index) * 2 + 1] = dd_mat_a[ij_index].val[1]
			ptr_td_mat_a[(ij_index) * 3]     = td_mat_a[ij_index].val[0]
			ptr_td_mat_a[(ij_index) * 3 + 1] = td_mat_a[ij_index].val[1]
			ptr_td_mat_a[(ij_index) * 3 + 2] = td_mat_a[ij_index].val[2]
			ptr_qd_mat_a[(ij_index) * 4]     = qd_mat_a[ij_index].val[0]
			ptr_qd_mat_a[(ij_index) * 4 + 1] = qd_mat_a[ij_index].val[1]
			ptr_qd_mat_a[(ij_index) * 4 + 2] = qd_mat_a[ij_index].val[2]
			ptr_qd_mat_a[(ij_index) * 4 + 3] = qd_mat_a[ij_index].val[3]

	# B := sqrt(3) * [max(i + 1, j + 1)]
	d_mat_b  = [ d_zero] * (mid_dim * col_dim)
	dd_mat_b = [dd_zero] * (mid_dim * col_dim)
	td_mat_b = [td_zero] * (mid_dim * col_dim)
	qd_mat_b = [qd_zero] * (mid_dim * col_dim)
	mp_mat_b = [mp_zero] * (mid_dim * col_dim)
	mpfr_mat_b = [mpfr_zero] * (mid_dim * col_dim)

	ptr_dd_mat_b = (ct.c_double * (mid_dim * col_dim * 2))()
	ptr_td_mat_b = (ct.c_double * (mid_dim * col_dim * 3))()
	ptr_qd_mat_b = (ct.c_double * (mid_dim * col_dim * 5))()

	ptr_dd_mat_c = (ct.c_double * (row_dim * col_dim * 2))()
	ptr_td_mat_c = (ct.c_double * (mid_dim * col_dim * 3))()
	ptr_qd_mat_c = (ct.c_double * (mid_dim * col_dim * 5))()

#	print('size_mat_b = ', len(dd_mat_b))
	for i in range(mid_dim):
		for j in range(col_dim):
			ij_index = i * col_dim + j
			if i > j:
				d_mat_b[ij_index]  = float(mpfr_sqrt3 * (i + 1))
				dd_mat_b[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt3 * (i + 1))
				td_mat_b[ij_index] = rdd.mpfr_get_td(mpfr_sqrt3 * (i + 1))
				qd_mat_b[ij_index] = rdd.mpfr_get_qd(mpfr_sqrt3 * (i + 1))
				mp_mat_b[ij_index] = mp_sqrt3 * mpmath.mp.mpf(i + 1)
				mpfr_mat_b[ij_index] = mpfr_sqrt3 * gmpy2.mpfr(i + 1)
			else:
				d_mat_b[ij_index]  = float(mpfr_sqrt3 * (j + 1))
				dd_mat_b[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt3 * (j + 1))
				td_mat_b[ij_index] = rdd.mpfr_get_td(mpfr_sqrt3 * (j + 1))
				qd_mat_b[ij_index] = rdd.mpfr_get_qd(mpfr_sqrt3 * (j + 1))
				mp_mat_b[ij_index] = mp_sqrt3 * mpmath.mp.mpf(j + 1)
				mpfr_mat_b[ij_index] = mpfr_sqrt3 * gmpy2.mpfr(j + 1)

			ptr_dd_mat_b[(ij_index) * 2]     = dd_mat_b[ij_index].val[0]
			ptr_dd_mat_b[(ij_index) * 2 + 1] = dd_mat_b[ij_index].val[1]
			ptr_td_mat_b[(ij_index) * 3]     = td_mat_b[ij_index].val[0]
			ptr_td_mat_b[(ij_index) * 3 + 1] = td_mat_b[ij_index].val[1]
			ptr_td_mat_b[(ij_index) * 3 + 2] = td_mat_b[ij_index].val[2]
			ptr_qd_mat_b[(ij_index) * 4]     = qd_mat_b[ij_index].val[0]
			ptr_qd_mat_b[(ij_index) * 4 + 1] = qd_mat_b[ij_index].val[1]
			ptr_qd_mat_b[(ij_index) * 4 + 2] = qd_mat_b[ij_index].val[2]
			ptr_qd_mat_b[(ij_index) * 4 + 3] = qd_mat_b[ij_index].val[3]

	print('double= ', d_mat_a[0] , d_mat_b[0], d_vec_a[0])
	print('dd    = ', dd_mat_a[0], dd_mat_b[0], dd_vec_a[0])
	print('td    = ', td_mat_a[0], td_mat_b[0], td_vec_a[0])
	print('qd    = ', qd_mat_a[0], qd_mat_b[0], qd_vec_a[0])
	print('mp    = ', mp_mat_a[0], mp_mat_b[0], mp_vec_a[0])
	print('mpfr  = ', mpfr_mat_a[0], mpfr_mat_b[0], mpfr_vec_a[0])

	print('p_dd  = ',
		rdd.dd_float(ptr_dd_mat_a[0], ptr_dd_mat_a[1]),
		rdd.dd_float(ptr_dd_mat_b[0], ptr_dd_mat_b[1]),
		rdd.dd_float(ptr_dd_vec_a[0], ptr_dd_vec_a[1])
	)
	print('p_td  = ', 
		rdd.td_float(ptr_td_mat_a[0], ptr_td_mat_a[1], ptr_td_mat_a[2]),
		rdd.td_float(ptr_td_mat_b[0], ptr_td_mat_b[1], ptr_td_mat_b[2]),
		rdd.td_float(ptr_td_vec_a[0], ptr_td_vec_a[1], ptr_td_vec_a[2])
	)
	print('p_qd  = ', 
		rdd.qd_float(ptr_qd_mat_a[0], ptr_qd_mat_a[1], ptr_qd_mat_a[2], ptr_qd_mat_a[3]),
		rdd.qd_float(ptr_qd_mat_b[0], ptr_qd_mat_b[1], ptr_qd_mat_b[2], ptr_qd_mat_b[3]),
		rdd.qd_float(ptr_qd_vec_a[0], ptr_qd_vec_a[1], ptr_qd_vec_a[2], ptr_qd_vec_a[3])
	)

	# ddvector
	ddvec_a = init_ddvector(mid_dim)
	set_ddvector_ddfloat(ddvec_a, ptr_dd_vec_a, mid_dim)
	ddvec_b = init_ddvector(row_dim)

	# ddmatrix
	ddmat_a = init_ddmatrix(row_dim, mid_dim)
	set_ddmatrix_ddfloat(ddmat_a, ptr_dd_mat_a, row_dim * mid_dim)
	ddmat_b = init_ddmatrix(mid_dim, col_dim)
	set_ddmatrix_ddfloat(ddmat_b, ptr_dd_mat_b, mid_dim * col_dim)
	ddmat_c = init_ddmatrix(row_dim, col_dim)
	#print(ddmat_a)
	#print('size a = ', ddmat_a.contents.row_dim, ddmat_a.contents.col_dim)
	#print('size b = ', ddmat_b.contents.row_dim, ddmat_b.contents.col_dim)
	#print('size c = ', ddmat_c.contents.row_dim, ddmat_c.contents.col_dim)

	# double
	start_time = time.time()
	d_mat_c = xd_mymatmul(d_mat_a, row_dim, mid_dim, d_mat_b, mid_dim, col_dim, d_zero)
	end_time = time.time()
	d_matmul_time = end_time - start_time
	print('d(', 53, ')')
	#cProfile.run('xd_mymatmul(d_mat_a, row_dim, mid_dim, d_mat_b, mid_dim, col_dim, d_zero)')

	# dd
	start_time = time.time()
	dd_mat_c = xd_mymatmul(dd_mat_a, row_dim, mid_dim, dd_mat_b, mid_dim, col_dim, dd_zero)
	end_time = time.time()
	dd_matmul_time = end_time - start_time
	print('dd(', 53 * 2, ')')
	#cProfile.run('xd_mymatmul(dd_mat_a, row_dim, mid_dim, dd_mat_b, mid_dim, col_dim, dd_zero)')

	# ptr_dd
	start_time = time.time()
#	ptr_dd_mat_c = rdd.dd_matmul_simple(ptr_dd_mat_a, row_dim, mid_dim, ptr_dd_mat_b, mid_dim, col_dim)
	rdd.dd_matmul_simple(ptr_dd_mat_c, row_dim, col_dim, ptr_dd_mat_a, row_dim, mid_dim, ptr_dd_mat_b, mid_dim, col_dim)
	end_time = time.time()
	ptr_dd_matmul_time = end_time - start_time

	# mat_ddmatrix_block
	start_time = time.time()
	mul_ddmatrix_block(ddmat_c, ddmat_a, ddmat_b, 32)
	end_time = time.time()
	dd_matmul_block_time = end_time - start_time
#	print('sq_dim = ', sq_dim, ' dd_mymatmul 計算時間(秒): ', end_time - start_time)

	# mat_ddmatrix_strassen
	start_time = time.time()
	mul_ddmatrix_strassen(ddmat_c, ddmat_a, ddmat_b, 32)
	end_time = time.time()
	dd_matmul_strassen_time = end_time - start_time

	# dd_mv
	start_time = time.time()
	dd_vec_b = xd_mymv(dd_mat_a, row_dim, mid_dim, dd_vec_a, mid_dim, dd_zero)
	end_time = time.time()
	dd_mymv_time = end_time - start_time

	# ptr_dd_mv
	start_time = time.time()
	rdd.dd_mvmul_simple(ptr_dd_vec_b, row_dim, ptr_dd_mat_a, row_dim, mid_dim, ptr_dd_vec_a, mid_dim)
	end_time = time.time()
	ptr_dd_mvmul_time = end_time - start_time

	# mat_ddmatrix_ddvec
	start_time = time.time()
	mul_ddmatrix_ddvec(ddvec_b, ddmat_a, ddvec_a)
	end_time = time.time()
	dd_mvmul_time = end_time - start_time

	# tdvector
	tdvec_a = init_tdvector(mid_dim)
	set_tdvector_tdfloat(tdvec_a, ptr_td_vec_a, mid_dim)
	tdvec_b = init_tdvector(row_dim)

	# tdmatrix
	tdmat_a = init_tdmatrix(row_dim, mid_dim)
	set_tdmatrix_tdfloat(tdmat_a, ptr_td_mat_a, row_dim * mid_dim)
	tdmat_b = init_tdmatrix(mid_dim, col_dim)
	set_tdmatrix_tdfloat(tdmat_b, ptr_td_mat_b, mid_dim * col_dim)
	tdmat_c = init_tdmatrix(row_dim, col_dim)

	# td
	start_time = time.time()
	td_mat_c = xd_mymatmul(td_mat_a, row_dim, mid_dim, td_mat_b, mid_dim, col_dim, td_zero)
	end_time = time.time()
	td_matmul_time = end_time - start_time
	print('td(', 53 * 3, ')')
	#cProfile.run('xd_mymatmul(td_mat_a, row_dim, mid_dim, td_mat_b, mid_dim, col_dim, td_zero)')
#	print('sq_dim = ', sq_dim, ' td_mymatmul 計算時間(秒): ', end_time - start_time)

	# ptr_td
	start_time = time.time()
#	ptr_td_mat_c = rdd.td_matmul_simple(ptr_td_mat_a, row_dim, mid_dim, ptr_td_mat_b, mid_dim, col_dim)
	rdd.td_matmul_simple(ptr_td_mat_c, row_dim, col_dim, ptr_td_mat_a, row_dim, mid_dim, ptr_td_mat_b, mid_dim, col_dim)
	end_time = time.time()
	ptr_td_matmul_time = end_time - start_time

	# mat_tdmatrix_block
	start_time = time.time()
	mul_tdmatrix_block(tdmat_c, tdmat_a, tdmat_b, 32)
	end_time = time.time()
	td_matmul_block_time = end_time - start_time
#	print('sq_dim = ', sq_dim, ' dd_mymatmul 計算時間(秒): ', end_time - start_time)

	# mat_tdmatrix_strassen
	start_time = time.time()
	mul_tdmatrix_strassen(tdmat_c, tdmat_a, tdmat_b, 32)
	end_time = time.time()
	td_matmul_strassen_time = end_time - start_time

	# td_mv
	start_time = time.time()
	td_vec_b = xd_mymv(td_mat_a, row_dim, mid_dim, td_vec_a, mid_dim, td_zero)
	end_time = time.time()
	td_mymv_time = end_time - start_time

	# ptr_td_mv
	start_time = time.time()
	rdd.td_mvmul_simple(ptr_td_vec_b, row_dim, ptr_td_mat_a, row_dim, mid_dim, ptr_td_vec_a, mid_dim)
	end_time = time.time()
	ptr_td_mvmul_time = end_time - start_time

	# mat_tdmatrix_tdvec
	start_time = time.time()
	mul_tdmatrix_tdvec(tdvec_b, tdmat_a, tdvec_a)
	end_time = time.time()
	td_mvmul_time = end_time - start_time

	# qdvector
	qdvec_a = init_qdvector(mid_dim)
	set_qdvector_qdfloat(qdvec_a, ptr_qd_vec_a, mid_dim)
	qdvec_b = init_qdvector(row_dim)

	# qdmatrix
	qdmat_a = init_qdmatrix(row_dim, mid_dim)
	set_qdmatrix_qdfloat(qdmat_a, ptr_qd_mat_a, row_dim * mid_dim)
	qdmat_b = init_qdmatrix(mid_dim, col_dim)
	set_qdmatrix_qdfloat(qdmat_b, ptr_qd_mat_b, mid_dim * col_dim)
	qdmat_c = init_qdmatrix(row_dim, col_dim)

	# qd
	start_time = time.time()
	qd_mat_c = xd_mymatmul(qd_mat_a, row_dim, mid_dim, qd_mat_b, mid_dim, col_dim, qd_zero)
	end_time = time.time()
	qd_matmul_time = end_time - start_time
	print('qd(', 53 * 4, ')')
	#cProfile.run('xd_mymatmul(qd_mat_a, row_dim, mid_dim, qd_mat_b, mid_dim, col_dim, qd_zero)')
#	print('sq_dim = ', sq_dim, ' qd_mymatmul 計算時間(秒): ', end_time - start_time)

	# ptr_qd
	start_time = time.time()
#	ptr_qd_mat_c = rdd.qd_matmul_simple(ptr_qd_mat_a, row_dim, mid_dim, ptr_qd_mat_b, mid_dim, col_dim)
	rdd.qd_matmul_simple(ptr_qd_mat_c, row_dim, col_dim, ptr_qd_mat_a, row_dim, mid_dim, ptr_qd_mat_b, mid_dim, col_dim)
	end_time = time.time()
	ptr_qd_matmul_time = end_time - start_time

	# mat_qdmatrix_block
	start_time = time.time()
	mul_qdmatrix_block(qdmat_c, qdmat_a, qdmat_b, 32)
	end_time = time.time()
	qd_matmul_block_time = end_time - start_time
#	print('sq_dim = ', sq_dim, ' dd_mymatmul 計算時間(秒): ', end_time - start_time)

	# mat_qdmatrix_strassen
	start_time = time.time()
	mul_qdmatrix_strassen(qdmat_c, qdmat_a, qdmat_b, 32)
	end_time = time.time()
	qd_matmul_strassen_time = end_time - start_time

	# qd_mv
	start_time = time.time()
	qd_vec_b = xd_mymv(qd_mat_a, row_dim, mid_dim, qd_vec_a, mid_dim, qd_zero)
	end_time = time.time()
	qd_mymv_time = end_time - start_time

	# ptr_qd_mv
	start_time = time.time()
	rdd.qd_mvmul_simple(ptr_qd_vec_b, row_dim, ptr_qd_mat_a, row_dim, mid_dim, ptr_qd_vec_a, mid_dim)
	end_time = time.time()
	ptr_qd_mvmul_time = end_time - start_time

	# mat_qdmatrix_qdvec
	start_time = time.time()
	mul_qdmatrix_qdvec(qdvec_b, qdmat_a, qdvec_a)
	end_time = time.time()
	qd_mvmul_time = end_time - start_time

	# mpmath
	start_time = time.time()
	mp_mat_c = xd_mymatmul(mp_mat_a, row_dim, mid_dim, mp_mat_b, mid_dim, col_dim, mp_zero)
	end_time = time.time()
	mp_matmul_time = end_time - start_time
	print('mp(', mpmath.mp.prec, ')')
	#cProfile.run('xd_mymatmul(mp_mat_a, row_dim, mid_dim, mp_mat_b, mid_dim, col_dim, mp_zero)')
#	print('sq_dim = ', sq_dim, ' qd_mymatmul 計算時間(秒): ', end_time - start_time)

	# mpfr
	start_time = time.time()
	mpfr_mat_c = xd_mymatmul(mpfr_mat_a, row_dim, mid_dim, mpfr_mat_b, mid_dim, col_dim, mpfr_zero)
	end_time = time.time()
	mpfr_matmul_time = end_time - start_time
	print('mpfr(', gmpy2.get_context().precision, ')')
	#Profile.run('xd_mymatmul(mpfr_mat_a, row_dim, mid_dim, mpfr_mat_b, mid_dim, col_dim, mpfr_zero)')
#	print('sq_dim = ', sq_dim, ' qd_mymatmul 計算時間(秒): ', end_time - start_time)

	print('sq_dim = ', sq_dim, f', d, dd, td, qd, mp({mpmath.mp.prec:5d}), mpfr({gmpy2.get_context().precision:5d}): {d_matmul_time:5.3f}, {dd_matmul_time:5.3f}, {td_matmul_time:5.3f}, {qd_matmul_time:5.3f}, {mp_matmul_time:5.3f}, {mpfr_matmul_time:5.3f}')
	print('sq_dim = ', sq_dim, f', ptr_dd, ptr_td, ptr_qd(sec): {ptr_dd_matmul_time:5.3f}, {ptr_td_matmul_time:5.3f}, {ptr_qd_matmul_time:5.3f}')
	print('sq_dim = ', sq_dim, f', ddmat , tdmat , qdmat(sec) : {dd_matmul_block_time:5.3f}, {td_matmul_block_time:5.3f}, {qd_matmul_block_time:5.3f}')
	print('sq_dim = ', sq_dim, f', ddmats, tdmats, qdmats(sec): {dd_matmul_strassen_time:5.3f}, {td_matmul_strassen_time:5.3f}, {qd_matmul_strassen_time:5.3f}')
	print('sq_dim = ', sq_dim, f',   myddvec,   mytdvec,   myqdvec(sec): {dd_mymv_time:5.3g}, {td_mymv_time:5.3g}, {qd_mymv_time:5.3g}')
	print('sq_dim = ', sq_dim, f', ptr_ddvec, ptr_tdvec, ptr_qdvec(sec): {ptr_dd_mvmul_time:5.3g}, {ptr_td_mvmul_time:5.3g}, {ptr_qd_mvmul_time:5.3g}')
	print('sq_dim = ', sq_dim, f',     ddvec,     tdvec,     qdvec(sec): {dd_mvmul_time:5.3g}, {td_mvmul_time:5.3g}, {qd_mvmul_time:5.3g}')


	print('double= ',  d_mat_c[0])
	print('dd    = ', dd_mat_c[0])
	print('td    = ', td_mat_c[0])
	print('qd    = ', qd_mat_c[0])
	#print('mp    = ', mp_mat_c[0])
	#print('mpfr  = ', mpfr_mat_c[0])
	print('ptr_dd= ', rdd.dd_float(ptr_dd_mat_c[0], ptr_dd_mat_c[1]))
	print('ptr_td= ', rdd.td_float(ptr_td_mat_c[0], ptr_td_mat_c[1], ptr_td_mat_c[2]))
	print('ptr_qd= ', rdd.qd_float(ptr_qd_mat_c[0], ptr_qd_mat_c[1], ptr_qd_mat_c[2], ptr_qd_mat_c[3]))
	print('ddmat = ', rdd.dd_float(ddmat_c.contents.element[0][0], ddmat_c.contents.element[1][0]))
	print('tdmat = ', rdd.td_float(tdmat_c.contents.element[0][0], tdmat_c.contents.element[1][0], tdmat_c.contents.element[2][0]))
	print('qdmat = ', rdd.qd_float(qdmat_c.contents.element[0][0], qdmat_c.contents.element[1][0], qdmat_c.contents.element[2][0], qdmat_c.contents.element[3][0]))

	#print('double= ',  d_mat_c[0])
	print('dd_vec = ', dd_vec_b[0])
	print('td_vec = ', td_vec_b[0])
	print('qd_vec = ', qd_vec_b[0])
	#print('mp    = ', mp_mat_c[0])
	#print('mpfr  = ', mpfr_mat_c[0])
	print('ptr_ddv= ', rdd.dd_float(ptr_dd_vec_b[0], ptr_dd_vec_b[1]))
	print('ptr_tdv= ', rdd.td_float(ptr_td_vec_b[0], ptr_td_vec_b[1], ptr_td_vec_b[2]))
	print('ptr_qdv= ', rdd.qd_float(ptr_td_vec_b[0], ptr_qd_vec_b[1], ptr_qd_vec_b[2], ptr_qd_vec_b[3]))
	print('ddvec  = ', rdd.dd_float(ddvec_b.contents.element[0][0], ddvec_b.contents.element[1][0]))
	print('tdvec  = ', rdd.td_float(tdvec_b.contents.element[0][0], tdvec_b.contents.element[1][0], tdvec_b.contents.element[2][0]))
	print('qdvec  = ', rdd.qd_float(qdvec_b.contents.element[0][0], qdvec_b.contents.element[1][0], qdvec_b.contents.element[2][0], qdvec_b.contents.element[3][0]))


	# delete
	del dd_vec_a, dd_vec_b, ptr_dd_vec_a, ptr_dd_vec_b
	del td_vec_a, td_vec_b, ptr_td_vec_a, ptr_td_vec_b
	del qd_vec_a, qd_vec_b, ptr_qd_vec_a, ptr_qd_vec_b
	free_ddvector(ddvec_a); free_ddvector(ddvec_b);
	free_tdvector(tdvec_a); free_tdvector(tdvec_b);
	free_qdvector(qdvec_a); free_qdvector(qdvec_b);

	del d_mat_a, d_mat_b, d_mat_c
	del dd_mat_a, dd_mat_b, dd_mat_c
	del td_mat_a, td_mat_b, td_mat_c
	del qd_mat_a, qd_mat_b, qd_mat_c
	#del mp_mat_a, mp_mat_b, mp_mat_c
	#del mpfr_mat_a, mpfr_mat_b, mpfr_mat_c
	del ptr_dd_mat_a, ptr_dd_mat_b, ptr_dd_mat_c
	del ptr_td_mat_a, ptr_td_mat_b, ptr_td_mat_c
	del ptr_qd_mat_a, ptr_qd_mat_b, ptr_qd_mat_c
	free_ddmatrix(ddmat_a); free_ddmatrix(ddmat_b); free_ddmatrix(ddmat_c);
	free_tdmatrix(tdmat_a); free_tdmatrix(tdmat_b); free_tdmatrix(tdmat_c);
	free_qdmatrix(qdmat_a); free_qdmatrix(qdmat_b); free_qdmatrix(qdmat_c);

librdd.rdd_end();

# ----
# BNCamPy.py: Base for Numerical Computation with Accelerated Multiple Precision arithmetic
# Copyright (c) 2021-2024 Tomonori Kouya, All right reserved.
# ----
# Version 0.0 , 2021-06-30 Started a simple implementation
# Version 0.01, 2024-03-17 Supported complex LU decomposition
# ----
# Prerequisite: mpmath, gmpy2
# rdd.py: DD(Double-Double), TD(Triple-Double), QD(Quadruple-Double) precision arithmetic
# rcdd.py: CDD(Complex DD), CTD(Complex TD), CQD(Complex QD) precision arithmetic
# BNCmatmul: Accelerated Basic Linear Algebra library: DD, TD, QD, MPFR(Arbitrary prec.)

import ctypes as ct
import rdd # dd_float, td_float, qd_float
import rcdd # cdd_float, ctd_float, cqd_float
import math
import gmpy2

import numpy as np
import mpmath
import time

# Profile
import cProfile

# RDD library
librdd = ct.cdll.LoadLibrary('librdd.so')

# RCDD library
librcdd = ct.cdll.LoadLibrary('librcdd.so')

# mpfr
libmpfr = ct.cdll.LoadLibrary('libmpfr.so')

# BNCmatmul
bncmatmul_ver = "0.22"
bncmatmul_prefix = 'libbncmatmul'
#bncmatmul_opt = ''
bncmatmul_opt = '_avx2'
#bncmatmul_opt = '_avx512'
#bncmatmul_opt = '_winograd'
#bncmatmul_opt = '_winograd_avx2'
#bncmatmul_opt = '_winograd_avx512'
#bncmatmul_opt = '-omp'
#bncmatmul_opt = '-omp_avx2'
#bncmatmul_opt = '-omp_avx512'
#bncmatmul_opt = '-omp_winograd'
#bncmatmul_opt = '-omp_winograd_avx2'
#bncmatmul_opt = '-omp_winograd_avx512'
bncmatmul_dllexp = 'so'
bncmatmul_libname = bncmatmul_prefix + '-' + bncmatmul_ver + bncmatmul_opt + '.' + bncmatmul_dllexp
#libbncmatmul = ct.cdll.LoadLibrary('libbncmatmul-' + bncmatmul_ver + '.so')
#libbncmatmul = ct.cdll.LoadLibrary('libbncmatmul-0.21_avx2.so')
libbncmatmul = ct.cdll.LoadLibrary(bncmatmul_libname)

print(libbncmatmul)

# set zeros for all types
d_zero  = 0.0
dd_zero = rdd.dd_float(0.0, 0.0)
td_zero = rdd.td_float(0.0, 0.0, 0.0)
qd_zero = rdd.qd_float(0.0, 0.0, 0.0, 0.0)
mp_zero = mpmath.mp.mpf(0)
mpfr_zero = gmpy2.mpfr(0)

cd_zero  = np.complex128(0.0)
cdd_zero = rcdd.cdd_float(0.0, 0.0, 0.0, 0.0)
ctd_zero = rcdd.ctd_float()
cqd_zero = rcdd.cqd_float()
mpc_zero = mpmath.mp.mpc(0, 0)


# create mpmath.vector
def init_mpvector(dim):
	ret_vec = mpmath.matrix([mp_zero for i in range(dim)])
	return ret_vec

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
init_ddvector.argtypes = [ct.c_long]
init_ddvector.restype = ct.POINTER(ddvector)

# free_ddvector
free_ddvector = libbncmatmul.free_ddvector
free_ddvector.argtypes = [ct.POINTER(ddvector)]
free_ddvector.restype = None

# dd array to ddvector
# DDVector vec -> ddfloat array
# void set_ddfloat_ddvec(ddfloat ret[], int ret_dim, DDVector vec);
set_ddfloat_ddvec = libbncmatmul.set_ddfloat_ddvec
set_ddfloat_ddvec.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ddvector)]
set_ddfloat_ddvec.restype = None

# ddfloat array -> DDVector ret
# void set_ddvector_ddfloat(DDVector ret, ddfloat array[], int array_dim);
set_ddvector_ddfloat = libbncmatmul.set_ddvector_ddfloat
set_ddvector_ddfloat.argtypes = [ct.POINTER(ddvector), ct.POINTER(ct.c_double), ct.c_int]
set_ddvector_ddfloat.restype = None

#void print_ddvector(DDVector vec)
print_ddvector = libbncmatmul.print_ddvector
print_ddvector.argtypes = [ct.POINTER(ddvector)]
print_ddvector.restype = None

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
init_ddmatrix.argtypes = [ct.c_long, ct.c_long]
init_ddmatrix.restype = ct.POINTER(ddmatrix)

# free_ddmatrix
free_ddmatrix = libbncmatmul.free_ddmatrix
free_ddmatrix.argtypes = [ct.POINTER(ddmatrix)]
free_ddmatrix.restype = None

# dd array to ddvector
# DDMatrix mat -> ddfloat array
# void set_ddfloat_ddmat(ddfloat ret[], int ret_dim, DDMatrix mat);
set_ddfloat_ddmat = libbncmatmul.set_ddfloat_ddmat
set_ddfloat_ddmat.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(ddmatrix)]
set_ddfloat_ddmat.restype = None

# ddfloat array -> DDmatrix ret
# void set_ddmatrix_ddfloat(DDMatrix ret, ddfloat array[], int array_dim);
set_ddmatrix_ddfloat = libbncmatmul.set_ddmatrix_ddfloat
set_ddmatrix_ddfloat.argtypes = [ct.POINTER(ddmatrix), ct.POINTER(ct.c_double), ct.c_int]
set_ddmatrix_ddfloat.restype = None

# Matrix-vector multiplicaiton
#void mul_ddmatrix_ddvec(DDVector ret, DDMatrix mat, DDVector vec);
mul_ddmatrix_ddvec = libbncmatmul.mul_ddmatrix_ddvec
mul_ddmatrix_ddvec.argtypes = [ct.POINTER(ddvector), ct.POINTER(ddmatrix), ct.POINTER(ddvector)]
mul_ddmatrix_ddvec.restype = None

# Block matrix multiplicaiton
#void mul_ddmatrix_block(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);
mul_ddmatrix_block = libbncmatmul.mul_ddmatrix_block
mul_ddmatrix_block.argtypes = [ct.POINTER(ddmatrix), ct.POINTER(ddmatrix), ct.POINTER(ddmatrix), ct.c_long]
mul_ddmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_ddmatrix_strassen(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);
mul_ddmatrix_strassen = libbncmatmul.mul_ddmatrix_strassen
mul_ddmatrix_strassen.argtypes = [ct.POINTER(ddmatrix), ct.POINTER(ddmatrix), ct.POINTER(ddmatrix), ct.c_long]
mul_ddmatrix_strassen.restype = None


# LU decomposition
#int DDLUdecompPM(DDMatrix a, long int ch[]);
DDLUdecompPM = libbncmatmul.DDLUdecompPM
DDLUdecompPM.argtypes = [ct.POINTER(ddmatrix), ct.POINTER(ct.c_long)]
DDLUdecompPM.restype = ct.c_int
DDLUdecompP = libbncmatmul.DDLUdecompP
DDLUdecompP.argtypes = [ct.POINTER(ddmatrix), ct.POINTER(ct.c_long)]
DDLUdecompP.restype = ct.c_int
#int SolveDDLSPM(DDVector answer, DDMatrix lu, DDVector b, long int ch[]);
SolveDDLSPM = libbncmatmul.SolveDDLSPM
SolveDDLSPM.argtypes = [ct.POINTER(ddvector), ct.POINTER(ddmatrix), ct.POINTER(ddvector), ct.POINTER(ct.c_long)]
SolveDDLSPM.restype = ct.c_int
SolveDDLSP = libbncmatmul.SolveDDLSP
SolveDDLSP.argtypes = [ct.POINTER(ddvector), ct.POINTER(ddmatrix), ct.POINTER(ddvector), ct.POINTER(ct.c_long)]
SolveDDLSP.restype = ct.c_int

#// cddfloat, ctdfloat, cqdfloat
#typedef struct { double val_re[DDSIZE]; double val_im[DDSIZE]; } cddfloat; // 53 * 2 = 106
#typedef struct { double val_re[TDSIZE]; double val_im[TDSIZE]; } ctdfloat; // 53 * 3 = 159
#typedef struct { double val_re[QDSIZE]; double val_im[QDSIZE]; } cqdfloat; // 53 * 4 = 212

# CDDVector class
# CDD vector
#typedef struct {
# 	DDVector re,
#	DDVector im
#} cddvector;
#typedef cddvector *CDDVector;
class cddvector(ct.Structure):
	_fields_ = [
		("re", ct.POINTER(ddvector)),
		("im", ct.POINTER(ddvector))
#		("re", ddvector),
#		("im", ddvector)
	]

# init_cddvector
init_cddvector = libbncmatmul.init_cddvector
init_cddvector.argtypes = [ct.c_long]
init_cddvector.restype = ct.POINTER(cddvector)

# free_cddvector
free_cddvector = libbncmatmul.free_cddvector
free_cddvector.argtypes = [ct.POINTER(cddvector)]
free_cddvector.restype = None

#// CDDVector vec -> ddfloat array
#void set_cddfloat_cddvec(cddfloat ret[], int ret_dim, CDDVector vec);
set_cddfloat_cddvec = libbncmatmul.set_cddfloat_cddvec
set_cddfloat_cddvec.argtypes = [ct.POINTER(rcdd.cdd_float), ct.c_int, ct.POINTER(cddvector)]
set_cddfloat_cddvec.restype = None

#// cddfloat array -> CDDVector ret
#void set_cddvector_cddfloat(CDDVector ret, cddfloat array[], int array_dim);
set_cddvector_cddfloat = libbncmatmul.set_cddvector_cddfloat
set_cddvector_cddfloat.argtypes = [ct.POINTER(cddvector), ct.POINTER(rcdd.cdd_float), ct.c_int]
set_cddvector_cddfloat.restype = None

#// ddvector -> cddvector
#void set_cddvector_ddvec(CDDVector ret, DDVector re_vec, DDVector im_vec);
set_cddvector_ddvec = libbncmatmul.set_cddvector_ddvec
set_cddvector_ddvec.argtypes = [ct.POINTER(cddvector), ct.POINTER(ddvector), ct.POINTER(ddvector)]
set_cddvector_ddvec.restype = None

# cdd_float -> CDDVector
def set_cddvector_cdd_float(ret, array):
	for i in range(len(array)):
		ret.contents.re.contents.element[0][i] = array[i].val_re[0]
		ret.contents.re.contents.element[1][i] = array[i].val_re[1]
		ret.contents.im.contents.element[0][i] = array[i].val_im[0]
		ret.contents.im.contents.element[1][i] = array[i].val_im[1]

# CDDVector -> cdd_float
def set_cdd_float_cddvector(vec): # (ret_array, vec):
	ret_array = []
	for i in range(vec.contents.re.contents.dim):
		#print(type(ret_array[i]))
		ret_array.append(rcdd.cdd_float(
			 vec.contents.re.contents.element[0][i],
			 vec.contents.re.contents.element[1][i],
			 vec.contents.im.contents.element[0][i],
			 vec.contents.im.contents.element[1][i],
		))
		#ret_array[i].val_re[0] = vec.contents.re.contents.element[0][i]
		#ret_array[i].val_re[1] = vec.contents.re.contents.element[1][i]
		#ret_array[i].val_im[0] = vec.contents.im.contents.element[0][i]
		#ret_array[i].val_im[1] = vec.contents.im.contents.element[1][i]
		#print(i, ' => ', ret_array[-1].val_re[0], ' + ', ret_array[-1].val_im[0], ' * I ')

	return ret_array

#// print cddvector
#void print_cddvector(CDDVector vec)
print_cddvector = libbncmatmul.print_cddvector
print_cddvector.argtypes = [ct.POINTER(cddvector)]
print_cddvector.restype = None

# CDDMatrix class
# CDD matrix
#typedef struct {
# 	DDMatrix re,
#	DDMatrix im
#} cddmatrix;
#typedef cddmatrix *CDDMatrix;
class cddmatrix(ct.Structure):
	_fields_ = [
		("re", ct.POINTER(ddmatrix)),
		("im", ct.POINTER(ddmatrix))
	]

# init_cddvector
init_cddmatrix = libbncmatmul.init_cddmatrix
init_cddmatrix.argtypes = [ct.c_long, ct.c_long]
init_cddmatrix.restype = ct.POINTER(cddmatrix)

# free_cddvector
free_cddmatrix = libbncmatmul.free_cddmatrix
free_cddmatrix.argtypes = [ct.POINTER(cddmatrix)]
free_cddmatrix.restype = None

#// ddmatrix -> cddmatrix
#void set_cddmatrix_ddmat(CDDMatrix ret, DDMatrix re_mat, DDMatrix im_mat);
set_cddmatrix_ddmat = libbncmatmul.set_cddmatrix_ddmat
set_cddmatrix_ddmat.argtypes = [ct.POINTER(cddmatrix), ct.POINTER(ddmatrix), ct.POINTER(ddmatrix)]
set_cddmatrix_ddmat.restype = None

# cddfloat array -> CDDmatrix ret
# void set_cddmatrix_cddfloat(CDDMatrix ret, cddfloat array[], int array_dim);
set_cddmatrix_cddfloat = libbncmatmul.set_cddmatrix_cddfloat
set_cddmatrix_cddfloat.argtypes = [ct.POINTER(cddmatrix), ct.POINTER(rcdd.cdd_float), ct.c_int]
set_cddmatrix_cddfloat.restype = None

# cdd_float -> CDDMatrix
def set_cddmatrix_cdd_float(ret, array):
	index = 0
	for i in range(ret.contents.re.contents.row_dim):
		for j in range(ret.contents.re.contents.col_dim):
			ij_index = i * ret.contents.re.contents.real_col_dim + j
			ret.contents.re.contents.element[0][ij_index] = array[index].val_re[0]
			ret.contents.re.contents.element[1][ij_index] = array[index].val_re[1]
			ret.contents.im.contents.element[0][ij_index] = array[index].val_im[0]
			ret.contents.im.contents.element[1][ij_index] = array[index].val_im[1]
			index += 1


# CDDMatrix -> cdd_float
def set_cdd_float_cddmatrix(array):
	index = 0
	ret = []
	for i in range(ret.contents.re.contents.row_dim):
		for j in range(ret.contents.re.contents.col_dim):
			ij_index = i * ret.contents.re.contents.real_col_dim + j
			#ret.contents.re.contents.element[0][ij_index] = array[index].val_re[0]
			#ret.contents.re.contents.element[1][ij_index] = array[index].val_re[1]
			#ret.contents.im.contents.element[0][ij_index] = array[index].val_im[0]
			#ret.contents.im.contents.element[1][ij_index] = array[index].val_im[1]
			ret.append(rcdd.cdd_float(
				ret.contents.re.contents.element[0][ij_index],
				ret.contents.re.contents.element[1][ij_index],
				ret.contents.im.contents.element[0][ij_index],
				ret.contents.im.contents.element[1][ij_index],
			))
			index += 1

	return ret

#// print cddmatrix
#void print_cddmatrix(CDDMatrix mat);
print_cddmatrix = libbncmatmul.print_cddmatrix
print_cddmatrix.argtypes = [ct.POINTER(cddmatrix)]
print_cddmatrix.restype = None

# Matrix-vector multiplicaiton
#void mul_cddmatrix_cddvec(CDDVector ret, CDDMatrix mat, CDDVector vec);
mul_cddmatrix_cddvec = libbncmatmul.mul_cddmatrix_cddvec_4m
#mul_cddmatrix_cddvec = libbncmatmul.mul_cddmatrix_cddvec_3m
mul_cddmatrix_cddvec.argtypes = [ct.POINTER(cddvector), ct.POINTER(cddmatrix), ct.POINTER(cddvector)]
mul_cddmatrix_cddvec.restype = None

# Block matrix multiplicaiton
#void mul_cddmatrix_block(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
mul_cddmatrix_block = libbncmatmul.mul_cddmatrix_block_3m
mul_cddmatrix_block.argtypes = [ct.POINTER(cddmatrix), ct.POINTER(cddmatrix), ct.POINTER(cddmatrix), ct.c_long]
mul_cddmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_cddmatrix_strassen(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
mul_cddmatrix_strassen = libbncmatmul.mul_cddmatrix_strassen_3m
mul_cddmatrix_strassen.argtypes = [ct.POINTER(cddmatrix), ct.POINTER(cddmatrix), ct.POINTER(cddmatrix), ct.c_long]
mul_cddmatrix_strassen.restype = None

# LU decomposition
#int CDDLUdecompPM(CDDMatrix a, long int ch[]);
CDDLUdecompPM = libbncmatmul.CDDLUdecompPM
CDDLUdecompPM.argtypes = [ct.POINTER(cddmatrix), ct.POINTER(ct.c_long)]
CDDLUdecompPM.restype = ct.c_int

CDDLUdecompP = libbncmatmul.CDDLUdecompP
CDDLUdecompP.argtypes = [ct.POINTER(cddmatrix), ct.POINTER(ct.c_long)]
CDDLUdecompP.restype = ct.c_int

#int SolveCDDLSPM(CDDVector answer, CDDMatrix lu, CDDVector b, long int ch[]);
SolveCDDLSPM = libbncmatmul.SolveCDDLSPM
SolveCDDLSPM.argtypes = [ct.POINTER(cddvector), ct.POINTER(cddmatrix), ct.POINTER(cddvector), ct.POINTER(ct.c_long)]
SolveCDDLSPM.restype = ct.c_int

SolveCDDLSP = libbncmatmul.SolveCDDLSP
SolveCDDLSP.argtypes = [ct.POINTER(cddvector), ct.POINTER(cddmatrix), ct.POINTER(cddvector), ct.POINTER(ct.c_long)]
SolveCDDLSP.restype = ct.c_int

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
init_tdvector.argtypes = [ct.c_long]
init_tdvector.restype = ct.POINTER(tdvector)

# free_tdvector
free_tdvector = libbncmatmul.free_tdvector
free_tdvector.argtypes = [ct.POINTER(tdvector)]
free_tdvector.restype = None

# dd array to tdvector
# TDVector vec -> tdfloat array
# void set_tdfloat_tdvec(tdfloat ret[], int ret_dim, TDVector vec);
set_tdfloat_tdvec = libbncmatmul.set_tdfloat_tdvec
set_tdfloat_tdvec.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(tdvector)]
set_tdfloat_tdvec.restype = None

# tdfloat array -> TDVector ret
# void set_tdvector_tdfloat(TDVector ret, tdfloat array[], int array_dim);
set_tdvector_tdfloat = libbncmatmul.set_tdvector_tdfloat
set_tdvector_tdfloat.argtypes = [ct.POINTER(tdvector), ct.POINTER(ct.c_double), ct.c_int]
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
init_tdmatrix.argtypes = [ct.c_long, ct.c_long]
init_tdmatrix.restype = ct.POINTER(tdmatrix)

# free_tdmatrix
free_tdmatrix = libbncmatmul.free_tdmatrix
free_tdmatrix.argtypes = [ct.POINTER(tdmatrix)]
free_tdmatrix.restype = None

# td array to tdvector
# TDMatrix mat -> tdfloat array
# void set_tdfloat_tdmat(tdfloat ret[], int ret_dim, TDMatrix mat);
set_tdfloat_tdmat = libbncmatmul.set_tdfloat_tdmat
set_tdfloat_tdmat.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(tdmatrix)]
set_tdfloat_tdmat.restype = None

# tdfloat array -> TDmatrix ret
# void set_tdmatrix_tdfloat(DDMatrix ret, tdfloat array[], int array_dim);
set_tdmatrix_tdfloat = libbncmatmul.set_tdmatrix_tdfloat
set_tdmatrix_tdfloat.argtypes = [ct.POINTER(tdmatrix), ct.POINTER(ct.c_double), ct.c_int]
set_tdmatrix_tdfloat.restype = None

# Matrix-vector multiplicaiton
#void mul_tdmatrix_tdvec(TDVector ret, TDMatrix mat, TDVector vec);
mul_tdmatrix_tdvec = libbncmatmul.mul_tdmatrix_tdvec
mul_tdmatrix_tdvec.argtypes = [ct.POINTER(tdvector), ct.POINTER(tdmatrix), ct.POINTER(tdvector)]
mul_tdmatrix_tdvec.restype = None

# Block matrix multiplicaiton
#void mul_tdmatrix_block(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);
mul_tdmatrix_block = libbncmatmul.mul_tdmatrix_block
mul_tdmatrix_block.argtypes = [ct.POINTER(tdmatrix), ct.POINTER(tdmatrix), ct.POINTER(tdmatrix), ct.c_long]
mul_tdmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_tdmatrix_strassen(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);
mul_tdmatrix_strassen = libbncmatmul.mul_tdmatrix_strassen
mul_tdmatrix_strassen.argtypes = [ct.POINTER(tdmatrix), ct.POINTER(tdmatrix), ct.POINTER(tdmatrix), ct.c_long]
mul_tdmatrix_strassen.restype = None

# LU decomposition
#int TDLUdecompPM(TDMatrix a, long int ch[]);
TDLUdecompPM = libbncmatmul.TDLUdecompPM
TDLUdecompPM.argtypes = [ct.POINTER(tdmatrix), ct.POINTER(ct.c_long)]
TDLUdecompPM.restype = ct.c_int
TDLUdecompP = libbncmatmul.TDLUdecompP
TDLUdecompP.argtypes = [ct.POINTER(tdmatrix), ct.POINTER(ct.c_long)]
TDLUdecompP.restype = ct.c_int
#int SolveDDLSPM(DDVector answer, DDMatrix lu, DDVector b, long int ch[]);
SolveTDLSPM = libbncmatmul.SolveTDLSPM
SolveTDLSPM.argtypes = [ct.POINTER(tdvector), ct.POINTER(tdmatrix), ct.POINTER(tdvector), ct.POINTER(ct.c_long)]
SolveTDLSPM.restype = ct.c_int
SolveTDLSP = libbncmatmul.SolveTDLSP
SolveTDLSP.argtypes = [ct.POINTER(tdvector), ct.POINTER(tdmatrix), ct.POINTER(tdvector), ct.POINTER(ct.c_long)]
SolveTDLSP.restype = ct.c_int

# CTDVector class
# CTD vector
#typedef struct {
# 	TDVector re,
#	TDVector im
#} ctdvector;
#typedef ctdvector *CTDVector;
class ctdvector(ct.Structure):
	_fields_ = [
		("re", ct.POINTER(tdvector)),
		("im", ct.POINTER(tdvector))
	]

# init_ctdvector
init_ctdvector = libbncmatmul.init_ctdvector
init_ctdvector.argtypes = [ct.c_long]
init_ctdvector.restype = ct.POINTER(ctdvector)

# free_ctdvector
free_ctdvector = libbncmatmul.free_ctdvector
free_ctdvector.argtypes = [ct.POINTER(ctdvector)]
free_ctdvector.restype = None

#// tdvector -> ctdvector
#void set_ctdvector_tdvec(CTDVector ret, TDVector re_vec, TDVector im_vec);
set_ctdvector_tdvec = libbncmatmul.set_ctdvector_tdvec
set_ctdvector_tdvec.argtypes = [ct.POINTER(ctdvector), ct.POINTER(tdvector), ct.POINTER(tdvector)]
set_ctdvector_tdvec.restype = None

# ctd_float -> CTDVector
def set_ctdvector_ctd_float(ret, array):
	for i in range(len(array)):
		ret.contents.re.contents.element[0][i] = array[i].val_re[0]
		ret.contents.re.contents.element[1][i] = array[i].val_re[1]
		ret.contents.re.contents.element[2][i] = array[i].val_re[2]
		ret.contents.im.contents.element[0][i] = array[i].val_im[0]
		ret.contents.im.contents.element[1][i] = array[i].val_im[1]
		ret.contents.im.contents.element[2][i] = array[i].val_im[2]


# CTDMatrix class
# CTD matrix
#typedef struct {
# 	TDMatrix re,
#	TDMatrix im
#} ctdmatrix;
#typedef ctdmatrix *CTDMatrix;
class ctdmatrix(ct.Structure):
	_fields_ = [
		("re", ct.POINTER(tdmatrix)),
		("im", ct.POINTER(tdmatrix))
	]

# init_ctdmatrix
init_ctdmatrix = libbncmatmul.init_ctdmatrix
init_ctdmatrix.argtypes = [ct.c_long, ct.c_long]
init_ctdmatrix.restype = ct.POINTER(ctdmatrix)

# free_ctdmatrix
free_ctdmatrix = libbncmatmul.free_ctdmatrix
free_ctdmatrix.argtypes = [ct.POINTER(ctdmatrix)]
free_ctdmatrix.restype = None

#// tdmatrix -> ctdmatrix
#void set_ctdmatrix_tdmat(CTDMatrix ret, TDMatrix re_mat, TDMatrix im_mat);
set_ctdmatrix_tdmat = libbncmatmul.set_ctdmatrix_tdmat
set_ctdmatrix_tdmat.argtypes = [ct.POINTER(ctdmatrix), ct.POINTER(tdmatrix), ct.POINTER(tdmatrix)]
set_ctdmatrix_tdmat.restype = None

# ctdfloat array -> CTDmatrix ret
# void set_ctdmatrix_ctdfloat(CTDMatrix ret, ctdfloat array[], int array_dim);
set_cddmatrix_ctdfloat = libbncmatmul.set_ctdmatrix_ctdfloat
set_cddmatrix_ctdfloat.argtypes = [ct.POINTER(ctdmatrix), ct.POINTER(rcdd.ctd_float), ct.c_int]
set_cddmatrix_ctdfloat.restype = None

# ctd_float -> CTDMatrix
def set_ctdmatrix_ctd_float(ret, array):
	index = 0
	for i in range(ret.contents.re.contents.row_dim):
		for j in range(ret.contents.re.contents.col_dim):
			ij_index = i * ret.contents.re.contents.real_col_dim + j
			ret.contents.re.contents.element[0][ij_index] = array[index].val_re[0]
			ret.contents.re.contents.element[1][ij_index] = array[index].val_re[1]
			ret.contents.re.contents.element[2][ij_index] = array[index].val_re[2]
			ret.contents.im.contents.element[0][ij_index] = array[index].val_im[0]
			ret.contents.im.contents.element[1][ij_index] = array[index].val_im[1]
			ret.contents.im.contents.element[2][ij_index] = array[index].val_im[2]
			index += 1


# Matrix-vector multiplicaiton
#void mul_ctdmatrix_ctdvec(CTDVector ret, CTDMatrix mat, CTDVector vec);
mul_ctdmatrix_ctdvec = libbncmatmul.mul_ctdmatrix_ctdvec_4m
#mul_cddmatrix_ctdvec = libbncmatmul.mul_ctdmatrix_ctdvec_3m
mul_ctdmatrix_ctdvec.argtypes = [ct.POINTER(ctdvector), ct.POINTER(ctdmatrix), ct.POINTER(ctdvector)]
mul_ctdmatrix_ctdvec.restype = None

# Block matrix multiplicaiton
#void mul_ctdmatrix_block(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
mul_ctdmatrix_block = libbncmatmul.mul_ctdmatrix_block_3m
mul_ctdmatrix_block.argtypes = [ct.POINTER(ctdmatrix), ct.POINTER(ctdmatrix), ct.POINTER(ctdmatrix), ct.c_long]
mul_ctdmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_ctdmatrix_strassen(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
mul_ctdmatrix_strassen = libbncmatmul.mul_ctdmatrix_strassen_3m
mul_ctdmatrix_strassen.argtypes = [ct.POINTER(ctdmatrix), ct.POINTER(ctdmatrix), ct.POINTER(ctdmatrix), ct.c_long]
mul_ctdmatrix_strassen.restype = None

# LU decomposition
#int CTDLUdecompPM(CTDMatrix a, long int ch[]);
CTDLUdecompPM = libbncmatmul.CTDLUdecompPM
CTDLUdecompPM.argtypes = [ct.POINTER(ctdmatrix), ct.POINTER(ct.c_long)]
CTDLUdecompPM.restype = ct.c_int
CTDLUdecompP = libbncmatmul.CTDLUdecompP
CTDLUdecompP.argtypes = [ct.POINTER(ctdmatrix), ct.POINTER(ct.c_long)]
CTDLUdecompP.restype = ct.c_int
#int SolveCTDLSPM(CDDVector answer, CTDMatrix lu, CTDVector b, long int ch[]);
SolveCTDLSPM = libbncmatmul.SolveCTDLSPM
SolveCTDLSPM.argtypes = [ct.POINTER(ctdvector), ct.POINTER(ctdmatrix), ct.POINTER(ctdvector), ct.POINTER(ct.c_long)]
SolveCTDLSPM.restype = ct.c_int
SolveCTDLSP = libbncmatmul.SolveCTDLSP
SolveCTDLSP.argtypes = [ct.POINTER(ctdvector), ct.POINTER(ctdmatrix), ct.POINTER(ctdvector), ct.POINTER(ct.c_long)]
SolveCTDLSP.restype = ct.c_int


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
init_qdvector.argtypes = [ct.c_long]
init_qdvector.restype = ct.POINTER(qdvector)

# free_qdvector
free_qdvector = libbncmatmul.free_qdvector
free_qdvector.argtypes = [ct.POINTER(qdvector)]
free_qdvector.restype = None

# qd array to qdvector
# QDVector vec -> qdfloat array
# void set_qdfloat_qdvec(qdfloat ret[], int ret_dim, QDVector vec);
set_qdfloat_qdvec = libbncmatmul.set_qdfloat_qdvec
set_qdfloat_qdvec.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(qdvector)]
set_qdfloat_qdvec.restype = None

# qdfloat array -> QDVector ret
# void set_qdvector_qdfloat(QDVector ret, qdfloat array[], int array_dim);
set_qdvector_qdfloat = libbncmatmul.set_qdvector_qdfloat
set_qdvector_qdfloat.argtypes = [ct.POINTER(qdvector), ct.POINTER(ct.c_double), ct.c_int]
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
init_qdmatrix.argtypes = [ct.c_long, ct.c_long]
init_qdmatrix.restype = ct.POINTER(qdmatrix)

# free_qdmatrix
free_qdmatrix = libbncmatmul.free_qdmatrix
free_qdmatrix.argtypes = [ct.POINTER(qdmatrix)]
free_qdmatrix.restype = None

# qd array to qdvector
# QDMatrix mat -> qdfloat array
# void set_qdfloat_qdmat(qdfloat ret[], int ret_dim, QDMatrix mat);
set_qdfloat_qdmat = libbncmatmul.set_qdfloat_qdmat
set_qdfloat_qdmat.argtypes = [ct.POINTER(ct.c_double), ct.c_int, ct.POINTER(qdmatrix)]
set_qdfloat_qdmat.restype = None

# qdfloat array -> QDmatrix ret
# void set_qdmatrix_qdfloat(QDMatrix ret, qdfloat array[], int array_dim);
set_qdmatrix_qdfloat = libbncmatmul.set_qdmatrix_qdfloat
set_qdmatrix_qdfloat.argtypes = [ct.POINTER(qdmatrix), ct.POINTER(ct.c_double), ct.c_int]
set_qdmatrix_qdfloat.restype = None

# Matrix-vector multiplicaiton
#void mul_qdmatrix_qdvec(QDVector ret, QDMatrix mat, QDVector vec);
mul_qdmatrix_qdvec = libbncmatmul.mul_qdmatrix_qdvec
mul_qdmatrix_qdvec.argtypes = [ct.POINTER(qdvector), ct.POINTER(qdmatrix), ct.POINTER(qdvector)]
mul_qdmatrix_qdvec.restype = None

# Block matrix multiplicaiton
#void mul_qdmatrix_block(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);
mul_qdmatrix_block = libbncmatmul.mul_qdmatrix_block
mul_qdmatrix_block.argtypes = [ct.POINTER(qdmatrix), ct.POINTER(qdmatrix), ct.POINTER(qdmatrix), ct.c_long]
mul_qdmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_qdmatrix_strassen(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);
mul_qdmatrix_strassen = libbncmatmul.mul_qdmatrix_strassen
mul_qdmatrix_strassen.argtypes = [ct.POINTER(qdmatrix), ct.POINTER(qdmatrix), ct.POINTER(qdmatrix), ct.c_long]
mul_qdmatrix_strassen.restype = None

# LU decomposition
#int QDLUdecompPM(QDMatrix a, long int ch[]);
QDLUdecompPM = libbncmatmul.QDLUdecompPM
QDLUdecompPM.argtypes = [ct.POINTER(qdmatrix), ct.POINTER(ct.c_long)]
QDLUdecompPM.restype = ct.c_int
QDLUdecompP = libbncmatmul.QDLUdecompP
QDLUdecompP.argtypes = [ct.POINTER(qdmatrix), ct.POINTER(ct.c_long)]
QDLUdecompP.restype = ct.c_int
#int SolveQDLSPM(QDVector answer, QDMatrix lu, QDVector b, long int ch[]);
SolveQDLSPM = libbncmatmul.SolveQDLSPM
SolveQDLSPM.argtypes = [ct.POINTER(qdvector), ct.POINTER(qdmatrix), ct.POINTER(qdvector), ct.POINTER(ct.c_long)]
SolveQDLSPM.restype = ct.c_int
SolveQDLSP = libbncmatmul.SolveQDLSP
SolveQDLSP.argtypes = [ct.POINTER(qdvector), ct.POINTER(qdmatrix), ct.POINTER(qdvector), ct.POINTER(ct.c_long)]
SolveQDLSP.restype = ct.c_int


# CQDVector class
# CQD vector
#typedef struct {
# 	QDVector re,
#	QDVector im
#} cqdvector;
#typedef cqdvector *CQDVector;
class cqdvector(ct.Structure):
	_fields_ = [
		("re", ct.POINTER(qdvector)),
		("im", ct.POINTER(qdvector))
	]

# init_cqdvector
init_cqdvector = libbncmatmul.init_cqdvector
init_cqdvector.argtypes = [ct.c_long]
init_cqdvector.restype = ct.POINTER(cqdvector)

# free_cqdvector
free_cqdvector = libbncmatmul.free_cqdvector
free_cqdvector.argtypes = [ct.POINTER(cqdvector)]
free_cqdvector.restype = None

#// qdvector -> cqdvector
#void set_cqdvector_qdvec(CQDVector ret, QDVector re_vec, QDVector im_vec);
set_cqdvector_qdvec = libbncmatmul.set_cqdvector_qdvec
set_cqdvector_qdvec.argtypes = [ct.POINTER(cqdvector), ct.POINTER(qdvector), ct.POINTER(qdvector)]
set_cqdvector_qdvec.restype = None

# cqd_float -> CQDVector
def set_cqdvector_cqd_float(ret, array):
	for i in range(len(array)):
		ret.contents.re.contents.element[0][i] = array[i].val_re[0]
		ret.contents.re.contents.element[1][i] = array[i].val_re[1]
		ret.contents.re.contents.element[2][i] = array[i].val_re[2]
		ret.contents.re.contents.element[3][i] = array[i].val_re[3]
		ret.contents.im.contents.element[0][i] = array[i].val_im[0]
		ret.contents.im.contents.element[1][i] = array[i].val_im[1]
		ret.contents.im.contents.element[2][i] = array[i].val_im[2]
		ret.contents.im.contents.element[3][i] = array[i].val_im[3]


# CQDMatrix class
# CQD matrix
#typedef struct {
# 	QDMatrix re,
#	QDMatrix im
#} cqdmatrix;
#typedef cqdmatrix *CQDMatrix;
class cqdmatrix(ct.Structure):
	_fields_ = [
		("re", ct.POINTER(qdmatrix)),
		("im", ct.POINTER(qdmatrix))
	]

# init_cqdmatrix
init_cqdmatrix = libbncmatmul.init_cqdmatrix
init_cqdmatrix.argtypes = [ct.c_long, ct.c_long]
init_cqdmatrix.restype = ct.POINTER(cqdmatrix)

# free_cqdmatrix
free_cqdmatrix = libbncmatmul.free_cqdmatrix
free_cqdmatrix.argtypes = [ct.POINTER(cqdmatrix)]
free_cqdmatrix.restype = None

#// qdmatrix -> cqdmatrix
#void set_cqdmatrix_qdmat(CQDMatrix ret, QDMatrix re_mat, QDMatrix im_mat);
set_cqdmatrix_qdmat = libbncmatmul.set_cqdmatrix_qdmat
set_cqdmatrix_qdmat.argtypes = [ct.POINTER(cqdmatrix), ct.POINTER(qdmatrix), ct.POINTER(qdmatrix)]
set_cqdmatrix_qdmat.restype = None

# cqd_float -> CQDMatrix
def set_cqdmatrix_cqd_float(ret, array):
	index = 0
	for i in range(ret.contents.re.contents.row_dim):
		for j in range(ret.contents.re.contents.col_dim):
			ij_index = i * ret.contents.re.contents.real_col_dim + j
			ret.contents.re.contents.element[0][ij_index] = array[index].val_re[0]
			ret.contents.re.contents.element[1][ij_index] = array[index].val_re[1]
			ret.contents.re.contents.element[2][ij_index] = array[index].val_re[2]
			ret.contents.re.contents.element[3][ij_index] = array[index].val_re[3]
			ret.contents.im.contents.element[0][ij_index] = array[index].val_im[0]
			ret.contents.im.contents.element[1][ij_index] = array[index].val_im[1]
			ret.contents.im.contents.element[2][ij_index] = array[index].val_im[2]
			ret.contents.im.contents.element[3][ij_index] = array[index].val_im[3]
			index += 1


# Matrix-vector multiplicaiton
#void mul_cqdmatrix_cqdvec(CQDVector ret, CQDMatrix mat, CQDVector vec);
mul_cqdmatrix_cqdvec = libbncmatmul.mul_cqdmatrix_cqdvec_4m
#mul_cddmatrix_ctdvec = libbncmatmul.mul_ctdmatrix_ctdvec_3m
mul_cqdmatrix_cqdvec.argtypes = [ct.POINTER(cqdvector), ct.POINTER(cqdmatrix), ct.POINTER(cqdvector)]
mul_cqdmatrix_cqdvec.restype = None

# Block matrix multiplicaiton
#void mul_cqdmatrix_block(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
mul_cqdmatrix_block = libbncmatmul.mul_cqdmatrix_block_3m
mul_cqdmatrix_block.argtypes = [ct.POINTER(cqdmatrix), ct.POINTER(cqdmatrix), ct.POINTER(cqdmatrix), ct.c_long]
mul_cqdmatrix_block.restype = None

# Strassen matrix multiplicaiton
#void mul_cqdmatrix_strassen(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
mul_cqdmatrix_strassen = libbncmatmul.mul_cqdmatrix_strassen_3m
mul_cqdmatrix_strassen.argtypes = [ct.POINTER(cqdmatrix), ct.POINTER(cqdmatrix), ct.POINTER(cqdmatrix), ct.c_long]
mul_cqdmatrix_strassen.restype = None

# LU decomposition
#int CQDLUdecompPM(CQDMatrix a, long int ch[]);
CQDLUdecompPM = libbncmatmul.CQDLUdecompPM
CQDLUdecompPM.argtypes = [ct.POINTER(cqdmatrix), ct.POINTER(ct.c_long)]
CQDLUdecompPM.restype = ct.c_int
CQDLUdecompP = libbncmatmul.CQDLUdecompP
CQDLUdecompP.argtypes = [ct.POINTER(cqdmatrix), ct.POINTER(ct.c_long)]
CQDLUdecompP.restype = ct.c_int
#int SolveCQDLSPM(CQDVector answer, CQDMatrix lu, CQDVector b, long int ch[]);
SolveCQDLSPM = libbncmatmul.SolveCQDLSPM
SolveCQDLSPM.argtypes = [ct.POINTER(cqdvector), ct.POINTER(cqdmatrix), ct.POINTER(cqdvector), ct.POINTER(ct.c_long)]
SolveCQDLSPM.restype = ct.c_int
SolveCQDLSP = libbncmatmul.SolveCQDLSP
SolveCQDLSP.argtypes = [ct.POINTER(cqdvector), ct.POINTER(cqdmatrix), ct.POINTER(cqdvector), ct.POINTER(ct.c_long)]
SolveCQDLSP.restype = ct.c_int

# MPFVector class
#typedef void            mpfr_void;
#typedef int             mpfr_int;
#typedef unsigned int    mpfr_uint;
#typedef long            mpfr_long;
#typedef unsigned long   mpfr_ulong;
#typedef size_t          mpfr_size_t;

#/* Definition of the main structure */
#typedef struct {
#  mpfr_prec_t  _mpfr_prec;
#  mpfr_sign_t  _mpfr_sign;
#  mpfr_exp_t   _mpfr_exp;
#  mp_limb_t   *_mpfr_d;
#typedef int          mpfr_sign_t;
#} __mpfr_struct;
class bnc_mpfr_struct(ct.Structure):
	_fields_ = [
		('_mpfr_prec', ct.c_ulong),
		('_mpfr_sign', ct.c_int),
		('_mpfr_exp', ct.c_long),
		('_mpfr_d', ct.POINTER(ct.c_ulong))
	]

# mpfr_init
mpfr_init = libmpfr.mpfr_init
mpfr_init.argtypes = [ct.POINTER(bnc_mpfr_struct)]
mpfr_init.restype = ct.c_int

# MPFR vector
#typedef struct{
#	unsigned long int prec;
#	mpf_t *element;
#	long int dim;
#	long int real_dim; // append in 2022-11-18(Fri) T.Kouya
#} mpfvector;
#typedef mpfvector *MPFVector;
class mpfvector(ct.Structure):
	_fields_ = [
		('prec', ct.c_ulong),
		('element', ct.POINTER(bnc_mpfr_struct)),
		('dim', ct.c_long),
		('real_dim', ct.c_long),
	]

# init_mpfvector
init_mpfvector = libbncmatmul.init_mpfvector
init_mpfvector.argtypes = [ct.c_long]
init_mpfvector.restype = ct.POINTER(mpfvector)

# free_cqdvector
free_mpfvector = libbncmatmul.free_mpfvector
free_mpfvector.argtypes = [ct.POINTER(mpfvector)]
free_mpfvector.restype = None

# MPFR matrix
#typedef struct{
#	unsigned long int prec;
#	mpf_t *element;
#	long int row_dim, col_dim;
#	long int real_row_dim, real_col_dim; // 2022-11-18(Fri) T.Kouya
#	void *element_block; // mantissa block
#} mpfmatrix;
#typedef mpfmatrix *MPFMatrix;
class mpfmatrix(ct.Structure):
	_fields_ = [
		('prec', ct.c_ulong),
		('element', ct.POINTER(bnc_mpfr_struct)),
		('row_dim', ct.c_long),
		('col_dim', ct.c_long),
		('real_row_dim', ct.c_long),
		('real_col_dim', ct.c_long),
		('element_block', ct.c_void_p)
	]

# init_mpfmatrix
init_mpfmatrix = libbncmatmul.init_mpfmatrix
init_mpfmatrix.argtypes = [ct.c_long, ct.c_long]
init_mpfmatrix.restype = ct.POINTER(mpfmatrix)

# free_mpfmatrix
free_mpfmatrix = libbncmatmul.free_mpfmatrix
free_mpfmatrix.argtypes = [ct.POINTER(mpfmatrix)]
free_mpfmatrix.restype = None

# set_mpfmatrix_ij
set_mpfmatrix_ij = libbncmatmul.set_mpfmatrix_ij
set_mpfmatrix_ij.argtypes = [ct.POINTER(mpfmatrix), ct.c_long, ct.c_long, ct.POINTER(bnc_mpfr_struct)]
set_mpfmatrix_ij.restype = None

#void set_mpfmatrix_ij_str(MPFMatrix mat, long int row_index, long int col_index, const char *str, int base);
set_mpfmatrix_ij_str = libbncmatmul.set_mpfmatrix_ij_str
set_mpfmatrix_ij_str.argtypes = [ct.POINTER(mpfmatrix), ct.c_long, ct.c_long, ct.c_char_p, ct.c_int]
set_mpfmatrix_ij_str.restype = None

# print_mpfmatrix
print_mpfmatrix = libbncmatmul.print_mpfmatrix
print_mpfmatrix.argtypes = [ct.POINTER(mpfmatrix)]
print_mpfmatrix.restype = None

# mul_mpfmatrix
mul_mpfmatrix = libbncmatmul.mul_mpfmatrix
mul_mpfmatrix.argtypes = [ct.POINTER(mpfmatrix), ct.POINTER(mpfmatrix), ct.POINTER(mpfmatrix)]
mul_mpfmatrix.restype = None

# normf
normf_mpfmatrix = libbncmatmul.normf_mpfmatrix
normf_mpfmatrix.argtypes = [ct.POINTER(bnc_mpfr_struct), ct.POINTER(mpfmatrix)]
normf_mpfmatrix.restype = None


# gmpy2.mpfr -> MPFMatrix
def set_mpfmatrix_mpfr(ret, array):
	buf = ct.create_string_buffer(gmpy2.get_context().real_prec * 2) # len(str(array[0])))
	index = 0
	for i in range(ret.contents.row_dim):
		for j in range(ret.contents.col_dim):
			ij_index = i * ret.contents.real_col_dim + j
			buf.value = str(array[ij_index]).encode('utf-8')
			set_mpfmatrix_ij_str(ret, i, j, buf.value, 10)
			index += 1


# c := A * b
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

# C := A * B
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

# Frobenius Norm
def xd_normf(mat_a, row_dim, col_dim, xd_zero):
	zero = xd_zero

	ret = xd_zero

	for i in range(0, row_dim):
		for j in range(0, col_dim):
			ij_index = i * col_dim + j
			ret = mat_a[ij_index] * mat_a[ij_index]

	ret = mpmath.sqrt(mpmath.mp.mpf(str(ret)))

	return ret

# mpmath.matrix to fpmatrix
def get_fpmatrix_mpmat(ret_fpmat, mpmat):
	row_dim, col_dim = mpmat.rows, mpmat.cols

	for i in range(row_dim):
		for j in range(col_dim):
			ret_fpmat[i, j] = mpmat[i, j]

	return

# fpmatrix to mpmath.matrix
def set_fpmatrix_mpmat(ret_mpmat, fpmat):
	row_dim, col_dim = mpmat.rows, mpmat.cols

	for i in range(row_dim):
		for j in range(col_dim):
			ret_mpmat[i, j] = mpmath.mpmathify(fpmat[i, j])

	return

# mpmath.vector to fpvector
def get_fpvector_mpvec(ret_fpvec, mpvec):
	dim = len(mpvec)

	for i in range(dim):
		ret_fpvec[i] = mpvec[i]

	return

# fpvector to mpmath.vector
def set_fpvector_mpvec(ret_mpvec, fpvec):
	dim = len(ret_mpvec)

	for i in range(dim):
		ret_mpvec[i] = mpmath.mpmathify(fpvec[i])

	return

# mpmath.matrix to ddmatrix
def get_ddmatrix_mpmat(ret_ddmat, mpmat):
	row_dim, col_dim = mpmat.rows, mpmat.cols
	dd_val = rdd.dd_float()
	real_col_dim = ret_ddmat.contents.real_col_dim

	for i in range(row_dim):
		for j in range(col_dim):
			ij_index = i * real_col_dim + j
			#print('i, j, ij_index = ', i, j, ij_index)
			dd_val = mpf_get_dd(mpmat[i, j])
			#set_ddmatrix_ij(ret_ddmat, i, j, dd_val)
			ret_ddmat.contents.element[0][ij_index] = dd_val.val[0]
			ret_ddmat.contents.element[1][ij_index] = dd_val.val[1]

	return ret_ddmat


# ddmatrix to mpmath.matrix
def set_ddmatrix_mpmat(ret_mpmat, ddmat):
	row_dim, col_dim = ret_mpmat.rows, ret_mpmat.cols
	dd_val = rdd.dd_float()
	real_col_dim = ddmat.contents.real_col_dim

	for i in range(row_dim):
		for j in range(col_dim):
			ij_index = i * real_col_dim + j
			dd_val = rdd.dd_float(
				ddmat.contents.element[0][ij_index],
				ddmat.contents.element[1][ij_index]
			)
			ret_mpmat[i, j] = mpf_set_dd(dd_val)

	return ret_mpmat

# mpmath.vector to ddvector
def get_ddvector_mpvec(ret_ddvec, mpvec):
	#print('rows, cols = ', mpvec.rows, mpvec.cols)
	row_dim = len(mpvec)
	dd_val = rdd.dd_float()

	for i in range(row_dim):
		dd_val = mpf_get_dd(mpvec[i, 0])
		#set_ddvector_ij(ret_ddvec, i, dd_val)
		ret_ddvec.contents.element[0][i] = dd_val.val[0]
		ret_ddvec.contents.element[1][i] = dd_val.val[1]

	return ret_ddvec

# ddvector to mpmath.vector
def set_ddvector_mpvec(ret_mpvec, ddvec):
	#row_dim = mpvec.rows
	row_dim = len(ret_mpvec)
	dd_val = rdd.dd_float()

	for i in range(row_dim):
		dd_val = rdd.dd_float(ddvec.contents.element[0][i], ddvec.contents.element[1][i])
		ret_mpvec[i] = mpf_set_dd(dd_val)

	return ret_mpvec

# mp_dd_print(mp, dd)
def mp_dd_print(mpvec, ddvec, decdigits = 32):
	for i in range(len(mpvec)):
		print(i, 
			mpmath.nstr(mpvec[i], decdigits), 
			mpmath.nstr(
				mpf_set_dd(
					rdd.dd_float(
						ddvec.contents.element[0][i],
						ddvec.contents.element[1][i]
					)
				),
			32)
		)
	
	return

# mpf_get_dd
def mpf_get_dd(mpf_val):
	dd_ret = rdd.dd_float()
	dd_ret.val[0] = float(mpf_val)
	dd_ret.val[1] = float(mpf_val - mpmath.mp.mpf(dd_ret.val[0]))
	return dd_ret

# mpf_set_dd
def mpf_set_dd(ddval, prec = None):
    old_prec = mpmath.mp.prec
    if prec != None:
        mpmath.mp.prec = prec

    r = mpmath.mp.mpf(0)
    with mpmath.workprec(128):
        mpf_val = [mpmath.mp.mpf(0), mpmath.mp.mpf(0)]
        mpf_val[0] = mpmath.mpmathify(ddval.val[0])
        mpf_val[1] = mpmath.mpmathify(ddval.val[1])

    r = mpf_val[0] + mpf_val[1]

    mpmath.mp.prec = old_prec

    return r

# mpc_get_cdd
# mpc -> cdd_float
def mpc_get_cdd_float(mpc_val):
	dd_ret_re = rdd.dd_float()
	dd_ret_im = rdd.dd_float()
	dd_ret_re = mpf_get_dd(mpc_val.real)
	dd_ret_im = mpf_get_dd(mpc_val.imag)
	#return rcdd.cdd_float(dd_ret_re, dd_ret_im)
	return rcdd.cdd_float(
		dd_ret_re.val[0], dd_ret_re.val[1],
		dd_ret_im.val[0], dd_ret_im.val[1]
	)

# cdd_float -> mpc
def mpc_set_cdd_float(cddval, prec = None):
    old_prec = mpmath.mp.prec
    if prec != None:
        mpmath.mp.prec = prec

    #r = mpmath.mp.mpc(0, 0)

    with mpmath.workprec(128):
        mpf_val_re = [mpmath.mp.mpf(0), mpmath.mp.mpf(0)]
        mpf_val_re[0] = mpmath.mpmathify(cddval.val_re[0])
        mpf_val_re[1] = mpmath.mpmathify(cddval.val_re[1])

    #r.real = mpf_val[0] + mpf_val[1]

    with mpmath.workprec(128):
        mpf_val_im = [mpmath.mp.mpf(0), mpmath.mp.mpf(0)]
        mpf_val_im[0] = mpmath.mpmathify(cddval.val_im[0])
        mpf_val_im[1] = mpmath.mpmathify(cddval.val_im[1])

	#r.imag = mpf_val[0] + mpf_val[1]
    r = mpmath.mp.mpc(
		mpf_val_re[0] + mpf_val_re[1],
		mpf_val_im[0] + mpf_val_im[1]
	)

    mpmath.mp.prec = old_prec

    return r

# mpf_get_td
def mpf_get_td(mpf_val):
    td_ret = rdd.td_float()
    td_ret.val[0] = float(mpf_val)
    td_ret.val[1] = float(mpf_val - mpmath.mp.mpf(td_ret.val[0]))
    td_ret.val[2] = float(mpf_val - mpmath.mp.mpf(td_ret.val[0]) - mpmath.mp.mpf(td_ret.val[1]))
    return td_ret

# mpf_set_td
def mpf_set_td(tdval, prec = None):
    old_prec = mpmath.mp.prec
    if prec != None:
        mpmath.mp.prec = prec
    r = mpmath.mp.mpf('0.0')
    mpf_val = [mpmath.mp.mpf('0.0'), mpmath.mp.mpf('0.0'), mpmath.mp.mpf('0.0')]
    mpf_val[0] = mpmath.mp.mpf(tdval.val[0])
    mpf_val[1] = mpmath.mp.mpf(tdval.val[1])
    mpf_val[2] = mpmath.mp.mpf(tdval.val[2])
    r = mpf_val[0] + mpf_val[1] + mpf_val[2]

    mpmath.mp.prec = old_prec

    return r

# mpc_get_ctd
# mpc -> ctd_float
def mpc_get_ctd_float(mpc_val):
	td_ret_re = rdd.td_float()
	td_ret_im = rdd.td_float()
	td_ret_re = mpf_get_td(mpc_val.real)
	td_ret_im = mpf_get_td(mpc_val.imag)
	#return rcdd.ctd_float(td_ret_re, td_ret_im)
	return rcdd.ctd_float(
		td_ret_re.val[0], td_ret_re.val[1], td_ret_re.val[2],
		td_ret_im.val[0], td_ret_im.val[1], td_ret_im.val[2]
	)

# ctd_float -> mpc
def mpc_set_ctd_float(ctdval, prec = None):
    old_prec = mpmath.mp.prec
    if prec != None:
        mpmath.mp.prec = prec

    #r = mpmath.mp.mpc(0, 0)

    with mpmath.workprec(192):
        mpf_val_re = [mpmath.mp.mpf(0), mpmath.mp.mpf(0), mpmath.mp.mpf(0)]
        mpf_val_re[0] = mpmath.mpmathify(ctdval.val_re[0])
        mpf_val_re[1] = mpmath.mpmathify(ctdval.val_re[1])
        mpf_val_re[2] = mpmath.mpmathify(ctdval.val_re[2])

    with mpmath.workprec(192):
        mpf_val_im = [mpmath.mp.mpf(0), mpmath.mp.mpf(0), mpmath.mp.mpf(0)]
        mpf_val_im[0] = mpmath.mpmathify(ctdval.val_im[0])
        mpf_val_im[1] = mpmath.mpmathify(ctdval.val_im[1])
        mpf_val_im[2] = mpmath.mpmathify(ctdval.val_im[2])

    #r.real = mpf_val[0] + mpf_val[1] + mpf_val[2]
	#r.imag = mpf_val[0] + mpf_val[1] + mpf_val[2]
    r = mpmath.mp.mpc(
		mpf_val_re[0] + mpf_val_re[1] + mpf_val_re[2],
		mpf_val_im[0] + mpf_val_im[1] + mpf_val_im[2]
	)

    mpmath.mp.prec = old_prec

    return r


# mpf_get_qd
def mpf_get_qd(mpf_val):
    qd_ret = rdd.qd_float()
    qd_ret.val[0] = float(mpf_val)
    qd_ret.val[1] = float(mpf_val - mpmath.mp.mpf(qd_ret.val[0]))
    qd_ret.val[2] = float(mpf_val - mpmath.mp.mpf(qd_ret.val[0]) - mpmath.mp.mpf(qd_ret.val[1]))
    qd_ret.val[3] = float(mpf_val - mpmath.mp.mpf(qd_ret.val[0]) - mpmath.mp.mpf(qd_ret.val[1]) - mpmath.mp.mpf(qd_ret.val[2]))
    return qd_ret

# mpf_set_qd
def mpf_set_qd(tdval, prec = None):
    old_prec = mpmath.mp.prec
    if prec != None:
        mpmath.mp.prec = prec
    r = mpmath.mp.mpf('0.0')
    mpf_val = [mpmath.mp.mpf('0.0'), mpmath.mp.mpf('0.0'), mpmath.mp.mpf('0.0'), mpmath.mp.mpf('0.0')]
    mpf_val[0] = mpmath.mp.mpf(tdval.val[0])
    mpf_val[1] = mpmath.mp.mpf(tdval.val[1])
    mpf_val[2] = mpmath.mp.mpf(tdval.val[2])
    mpf_val[3] = mpmath.mp.mpf(tdval.val[3])
    r = mpf_val[0] + mpf_val[1] + mpf_val[2] + mpf_val[3]

    mpmath.mp.prec = old_prec

    return r

# mpc_get_cqd
# mpc -> cqd_float
def mpc_get_cqd_float(mpc_val):
	qd_ret_re = rdd.qd_float()
	qd_ret_im = rdd.qd_float()
	qd_ret_re = mpf_get_qd(mpc_val.real)
	qd_ret_im = mpf_get_qd(mpc_val.imag)
	#return rcdd.cqd_float(qd_ret_re, qd_ret_im)
	return rcdd.cqd_float(
		qd_ret_re.val[0], qd_ret_re.val[1], qd_ret_re.val[2], qd_ret_re.val[3],
		qd_ret_im.val[0], qd_ret_im.val[1], qd_ret_im.val[2], qd_ret_im.val[3]
	)

# cqd_float -> mpc
def mpc_set_cqd_float(cqdval, prec = None):
    old_prec = mpmath.mp.prec
    if prec != None:
        mpmath.mp.prec = prec

    #r = mpmath.mp.mpc(0, 0)

    with mpmath.workprec(256):
        mpf_val_re = [mpmath.mp.mpf(0), mpmath.mp.mpf(0), mpmath.mp.mpf(0), mpmath.mp.mpf(0)]
        mpf_val_re[0] = mpmath.mpmathify(cqdval.val_re[0])
        mpf_val_re[1] = mpmath.mpmathify(cqdval.val_re[1])
        mpf_val_re[2] = mpmath.mpmathify(cqdval.val_re[2])
        mpf_val_re[3] = mpmath.mpmathify(cqdval.val_re[3])

    with mpmath.workprec(256):
        mpf_val_im = [mpmath.mp.mpf(0), mpmath.mp.mpf(0), mpmath.mp.mpf(0), mpmath.mp.mpf(0)]
        mpf_val_im[0] = mpmath.mpmathify(cqdval.val_im[0])
        mpf_val_im[1] = mpmath.mpmathify(cqdval.val_im[1])
        mpf_val_im[2] = mpmath.mpmathify(cqdval.val_im[2])
        mpf_val_im[2] = mpmath.mpmathify(cqdval.val_im[2])

    #r.real = mpf_val[0] + mpf_val[1] + mpf_val[2] + mpf_val[3]
	#r.imag = mpf_val[0] + mpf_val[1] + mpf_val[2] + mpf_val[3]
    r = mpmath.mp.mpc(
		mpf_val_re[0] + mpf_val_re[1] + mpf_val_re[2] + mpf_val_re[3],
		mpf_val_im[0] + mpf_val_im[1] + mpf_val_im[2] + mpf_val_im[3]
	)

    mpmath.mp.prec = old_prec

    return r


# mpmath.matrix to tdmatrix
def get_tdmatrix_mpmat(ret_tdmat, mpmat):
	row_dim, col_dim = mpmat.rows, mpmat.cols
	td_val = rdd.td_float()
	real_col_dim = ret_tdmat.contents.real_col_dim

	for i in range(row_dim):
		for j in range(col_dim):
			ij_index = i * real_col_dim + j
			#print('i, j, ij_index = ', i, j, ij_index)
			td_val = mpf_get_td(mpmat[i, j])
			#set_ddmatrix_ij(ret_ddmat, i, j, dd_val)
			ret_tdmat.contents.element[0][ij_index] = td_val.val[0]
			ret_tdmat.contents.element[1][ij_index] = td_val.val[1]
			ret_tdmat.contents.element[2][ij_index] = td_val.val[2]

	return ret_tdmat


# tdmatrix to mpmath.matrix
def set_tdmatrix_mpmat(ret_mpmat, tdmat):
	row_dim, col_dim = ret_mpmat.rows, ret_mpmat.cols
	td_val = rdd.td_float()
	real_col_dim = tdmat.contents.real_col_dim

	for i in range(row_dim):
		for j in range(col_dim):
			ij_index = i * real_col_dim + j
			td_val = rdd.td_float(
				tdmat.contents.element[0][ij_index],
				tdmat.contents.element[1][ij_index],
				tdmat.contents.element[2][ij_index]
			)
			ret_mpmat[i, j] = mpf_set_td(td_val)

	return ret_mpmat

# mpmath.vector to tdvector
def get_tdvector_mpvec(ret_tdvec, mpvec):
	#print('rows, cols = ', mpvec.rows, mpvec.cols)
	row_dim = len(mpvec)
	td_val = rdd.td_float()

	for i in range(row_dim):
		td_val = mpf_get_td(mpvec[i, 0])
		#set_ddvector_ij(ret_ddvec, i, dd_val)
		ret_tdvec.contents.element[0][i] = td_val.val[0]
		ret_tdvec.contents.element[1][i] = td_val.val[1]
		ret_tdvec.contents.element[2][i] = td_val.val[2]

	return ret_tdvec

# tdvector to mpmath.vector
def set_tdvector_mpvec(ret_mpvec, tdvec):
	#row_dim = mpvec.rows
	row_dim = len(ret_mpvec)
	td_val = rdd.td_float()

	for i in range(row_dim):
		td_val = rdd.td_float(
			tdvec.contents.element[0][i],
			tdvec.contents.element[1][i],
			tdvec.contents.element[2][i]
		)
		ret_mpvec[i] = mpf_set_td(td_val)

	return ret_mpvec

# mpmath.matrix to qdmatrix
def get_qdmatrix_mpmat(ret_qdmat, mpmat):
	row_dim, col_dim = mpmat.rows, mpmat.cols
	qd_val = rdd.qd_float()
	real_col_dim = ret_qdmat.contents.real_col_dim

	for i in range(row_dim):
		for j in range(col_dim):
			ij_index = i * real_col_dim + j
			#print('i, j, ij_index = ', i, j, ij_index)
			qd_val = mpf_get_qd(mpmat[i, j])
			#set_ddmatrix_ij(ret_ddmat, i, j, dd_val)
			ret_qdmat.contents.element[0][ij_index] = qd_val.val[0]
			ret_qdmat.contents.element[1][ij_index] = qd_val.val[1]
			ret_qdmat.contents.element[2][ij_index] = qd_val.val[2]
			ret_qdmat.contents.element[3][ij_index] = qd_val.val[3]

	return ret_qdmat


# qdmatrix to mpmath.matrix
def set_qdmatrix_mpmat(ret_mpmat, qdmat):
	row_dim, col_dim = ret_mpmat.rows, ret_mpmat.cols
	qd_val = rdd.qd_float()
	real_col_dim = qdmat.contents.real_col_dim

	for i in range(row_dim):
		for j in range(col_dim):
			ij_index = i * real_col_dim + j
			qd_val = rdd.qd_float(
				qdmat.contents.element[0][ij_index],
				qdmat.contents.element[1][ij_index],
				qdmat.contents.element[2][ij_index],
				qdmat.contents.element[3][ij_index]
			)
			ret_mpmat[i, j] = mpf_set_qd(qd_val)

	return ret_mpmat

# mpmath.vector to qdvector
def get_qdvector_mpvec(ret_qdvec, mpvec):
	#print('rows, cols = ', mpvec.rows, mpvec.cols)
	row_dim = len(mpvec)
	qd_val = rdd.qd_float()

	for i in range(row_dim):
		qd_val = mpf_get_qd(mpvec[i, 0])
		#set_ddvector_ij(ret_ddvec, i, dd_val)
		ret_qdvec.contents.element[0][i] = qd_val.val[0]
		ret_qdvec.contents.element[1][i] = qd_val.val[1]
		ret_qdvec.contents.element[2][i] = qd_val.val[2]
		ret_qdvec.contents.element[3][i] = qd_val.val[3]

	return ret_qdvec

# qdvector to mpmath.vector
def set_qdvector_mpvec(ret_mpvec, qdvec):
	#row_dim = mpvec.rows
	row_dim = len(ret_mpvec)
	qd_val = rdd.qd_float()

	for i in range(row_dim):
		qd_val = rdd.qd_float(
			qdvec.contents.element[0][i],
			qdvec.contents.element[1][i],
			qdvec.contents.element[2][i],
			qdvec.contents.element[3][i]
		)
		ret_mpvec[i] = mpf_set_qd(qd_val)

	return ret_mpvec

# interative refinement with single & mpf_t mixed precision arithmetic
# solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
# unsigned long long_prec, short_prec; // long_prec > short_prec
def iterative_refinement_dd_mp(a, b, maxtimes, rtol, atol): #rtol = 1.0e-30, atol = 0.0):

	#int i, itimes;
	#int *af_ch;
	dim = b.rows

	print('dim = ', dim)
	print('prec = ', mpmath.mp.prec)

	# Initialize
	af_ch = (ct.c_long * (dim))()
	af = init_ddmatrix(dim, dim)
	bf = init_ddvector(dim)
	xf = init_ddvector(dim)

	x = init_mpvector(dim)
	res = init_mpvector(dim)

	resf = init_ddvector(dim)
	z = init_mpvector(dim)
	zf = init_ddvector(dim)
	#af_ch = (ct.c_long * (dim))
	#print('z.length = ', len(z))

	# norm_a := ||A||_F
	norm_a = mpmath.mnorm(a, 'fro')

	print('||A||_F = ', mpmath.nstr(norm_a, 10))
	# Make short precision copy of A and b
	#set_array(af, a, dim * dim);
	af = get_ddmatrix_mpmat(af, a)
	#print('Convert A - A^(S)')
	#print(af)

	#set_array(bf, b, dim);
	bf = get_ddvector_mpvec(bf, b)
	#print('Convert b - b^(S)')
	#mp_dd_print(b, bf)

	# Compute LU factorization in short precision
	# LU decomposition with partial pivoting
	#LU<S>(af, dim, af_ch);
	DDLUdecompPM(af, af_ch)
	#DDLUdecompP(af, af_ch)
	#print('LU ')

	# Apply back-solve in short precision with short precision factors
	#solve_LU_linear_eq<S>(xf, af, bf, dim, af_ch);
	SolveDDLSPM(xf, af, bf, af_ch)
	#SolveDDLSP(xf, af, bf, af_ch)

	# Promote te solution from short precision to long precision
	#set_array(x, xf, dim);
	x = set_ddvector_mpvec(x, xf)
	print('x, xf = ')
	#mp_dd_print(x, xf)

	print('iterative_ref start!')
	# repeat iterative refinement process
	for itimes in range(maxtimes):
		# Compute residual in long precision
		# res = b - a * x
		#mymv<L>(z, a, x, dim);
		#myaxpy<L>(res, (L)(-1), z, b, dim);
		#res = mpmath.residual(a, x, b)
		res = b - a * x
		#print(res)
	

		# until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		#norm_x = mynorm2<L>(x, dim);
		norm_x = mpmath.norm(x)
		#norm_res = mynorm2<L>(res, dim);
		norm_res = mpmath.norm(res)
		#print(itimes, norm_x, norm_res)

		if norm_res < mpmath.sqrt(mpmath.mpf(dim)) * rtol * norm_a * norm_x + atol:
			break

		# normalization: res := coef * res
		#normalization_coef = (L)1 / norm_res;
		normalization_coef = mpmath.mpmathify(1) / norm_res
		#myscal<L>(res, normalization_coef, res, dim);
		res = normalization_coef * res

		# Demote the residual from long precision to short precison
		#set_array(resf, res, dim);
		resf = get_ddvector_mpvec(resf, res)

		# Back-solve on short precision residual and short precision factors
		#solve_LU_linear_eq<S>(zf, af, resf, dim, af_ch);
		SolveDDLSPM(zf, af, resf, af_ch)
		#SolveDDLSP(zf, af, resf, af_ch)

		# Promote the correction from short precision to long precision
		#set_array(z, zf, dim);
		z = set_ddvector_mpvec(z, zf)

		# reverse normalizationi
		#myscal<L>(z, norm_res, z, dim);
		z = norm_res * z

		# Update solution in long precision
		#myaxpy<L>(x, (L)1, x, z, dim);
		#print(x, x+z)
		x = x + z
		#print(x)

		# for debug
		#std::cout << itimes << ", " << std::setprecision(10) << norm_res << std::endl;
		print(f'{itimes:4d}, {mpmath.nstr(norm_res, 10):s}, {mpmath.nstr(mpmath.norm(z), 10):s}')


	# if fail, retry in mpf_t precision
	if itimes >= maxtimes:
		# mpf_t precision
		#LU<L>(a, dim, af_ch);
		# SolveMPFLS(x, a, b);
		#solve_LU_linear_eq<L>(x, a, b, dim, af_ch);
		x = mpmath.lu_solve(a, b)

	#x = mpmath.lu_solve(a, b)

	# Clear
	free_ddmatrix(af)
	free_ddvector(bf)
	free_ddvector(xf)
	free_ddvector(resf)
	del res
	del zf
	del z
	del af_ch

	return x, itimes

# interative refinement with single & mpf_t mixed precision arithmetic
# solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
# unsigned long long_prec, short_prec; // long_prec > short_prec
def iterative_refinement_td_mp(a, b, maxtimes, rtol, atol): #rtol = 1.0e-30, atol = 0.0):

	#int i, itimes;
	#int *af_ch;
	dim = b.rows

	print('dim = ', dim)
	print('prec = ', mpmath.mp.prec)

	# Initialize
	af_ch = (ct.c_long * (dim))()
	af = init_tdmatrix(dim, dim)
	bf = init_tdvector(dim)
	xf = init_tdvector(dim)

	x = init_mpvector(dim)
	res = init_mpvector(dim)

	resf = init_tdvector(dim)
	z = init_mpvector(dim)
	zf = init_tdvector(dim)
	#af_ch = (ct.c_long * (dim))
	#print('z.length = ', len(z))

	# norm_a := ||A||_F
	norm_a = mpmath.mnorm(a, 'fro')

	print('||A||_F = ', mpmath.nstr(norm_a, 10))
	# Make short precision copy of A and b
	#set_array(af, a, dim * dim);
	af = get_tdmatrix_mpmat(af, a)
	#print('Convert A - A^(S)')
	#print(af)

	#set_array(bf, b, dim);
	bf = get_tdvector_mpvec(bf, b)
	#print('Convert b - b^(S)')
	#mp_dd_print(b, bf)

	# Compute LU factorization in short precision
	# LU decomposition with partial pivoting
	#LU<S>(af, dim, af_ch);
	TDLUdecompPM(af, af_ch)
	#TDLUdecompP(af, af_ch)
	#print('LU ')

	# Apply back-solve in short precision with short precision factors
	#solve_LU_linear_eq<S>(xf, af, bf, dim, af_ch);
	SolveTDLSPM(xf, af, bf, af_ch)
	#SolveTDLSP(xf, af, bf, af_ch)

	# Promote te solution from short precision to long precision
	#set_array(x, xf, dim);
	x = set_tdvector_mpvec(x, xf)
	print('x, xf = ')
	#mp_dd_print(x, xf)

	print('iterative_ref start!')
	# repeat iterative refinement process
	for itimes in range(maxtimes):
		# Compute residual in long precision
		# res = b - a * x
		#mymv<L>(z, a, x, dim);
		#myaxpy<L>(res, (L)(-1), z, b, dim);
		#res = mpmath.residual(a, x, b)
		res = b - a * x
		#print(res)
	

		# until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		#norm_x = mynorm2<L>(x, dim);
		norm_x = mpmath.norm(x)
		#norm_res = mynorm2<L>(res, dim);
		norm_res = mpmath.norm(res)
		#print(itimes, norm_x, norm_res)

		if norm_res < mpmath.sqrt(mpmath.mpf(dim)) * rtol * norm_a * norm_x + atol:
			break

		# normalization: res := coef * res
		#normalization_coef = (L)1 / norm_res;
		normalization_coef = mpmath.mpmathify(1) / norm_res
		#myscal<L>(res, normalization_coef, res, dim);
		res = normalization_coef * res

		# Demote the residual from long precision to short precison
		#set_array(resf, res, dim);
		resf = get_tdvector_mpvec(resf, res)

		# Back-solve on short precision residual and short precision factors
		#solve_LU_linear_eq<S>(zf, af, resf, dim, af_ch);
		SolveTDLSPM(zf, af, resf, af_ch)
		#SolveTDLSP(zf, af, resf, af_ch)

		# Promote the correction from short precision to long precision
		#set_array(z, zf, dim);
		z = set_tdvector_mpvec(z, zf)

		# reverse normalizationi
		#myscal<L>(z, norm_res, z, dim);
		z = norm_res * z

		# Update solution in long precision
		#myaxpy<L>(x, (L)1, x, z, dim);
		#print(x, x+z)
		x = x + z
		#print(x)

		# for debug
		#std::cout << itimes << ", " << std::setprecision(10) << norm_res << std::endl;
		print(f'{itimes:4d}, {mpmath.nstr(norm_res, 10):s}, {mpmath.nstr(mpmath.norm(z), 10):s}')


	# if fail, retry in mpf_t precision
	if itimes >= maxtimes:
		# mpf_t precision
		#LU<L>(a, dim, af_ch);
		# SolveMPFLS(x, a, b);
		#solve_LU_linear_eq<L>(x, a, b, dim, af_ch);
		x = mpmath.lu_solve(a, b)

	#x = mpmath.lu_solve(a, b)

	# Clear
	free_tdmatrix(af)
	free_tdvector(bf)
	free_tdvector(xf)
	free_tdvector(resf)
	del res
	del zf
	del z
	del af_ch

	return x, itimes

# interative refinement with single & mpf_t mixed precision arithmetic
# solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
# unsigned long long_prec, short_prec; // long_prec > short_prec
def iterative_refinement_qd_mp(a, b, maxtimes, rtol, atol): #rtol = 1.0e-30, atol = 0.0):

	#int i, itimes;
	#int *af_ch;
	dim = b.rows

	print('dim = ', dim)
	print('prec = ', mpmath.mp.prec)

	# Initialize
	af_ch = (ct.c_long * (dim))()
	af = init_qdmatrix(dim, dim)
	bf = init_qdvector(dim)
	xf = init_qdvector(dim)

	x = init_mpvector(dim)
	res = init_mpvector(dim)

	resf = init_qdvector(dim)
	z = init_mpvector(dim)
	zf = init_qdvector(dim)
	#af_ch = (ct.c_long * (dim))
	#print('z.length = ', len(z))

	# norm_a := ||A||_F
	norm_a = mpmath.mnorm(a, 'fro')

	print('||A||_F = ', mpmath.nstr(norm_a, 10))
	# Make short precision copy of A and b
	#set_array(af, a, dim * dim);
	af = get_qdmatrix_mpmat(af, a)
	#print('Convert A - A^(S)')
	#print(af)

	#set_array(bf, b, dim);
	bf = get_qdvector_mpvec(bf, b)
	#print('Convert b - b^(S)')
	#mp_dd_print(b, bf)

	# Compute LU factorization in short precision
	# LU decomposition with partial pivoting
	#LU<S>(af, dim, af_ch);
	QDLUdecompPM(af, af_ch)
	#QDLUdecompP(af, af_ch)
	#print('LU ')

	# Apply back-solve in short precision with short precision factors
	#solve_LU_linear_eq<S>(xf, af, bf, dim, af_ch);
	SolveQDLSPM(xf, af, bf, af_ch)
	#SolveQDLSP(xf, af, bf, af_ch)

	# Promote te solution from short precision to long precision
	#set_array(x, xf, dim);
	x = set_qdvector_mpvec(x, xf)
	print('x, xf = ')
	#mp_dd_print(x, xf)

	print('iterative_ref start!')
	# repeat iterative refinement process
	for itimes in range(maxtimes):
		# Compute residual in long precision
		# res = b - a * x
		#mymv<L>(z, a, x, dim);
		#myaxpy<L>(res, (L)(-1), z, b, dim);
		#res = mpmath.residual(a, x, b)
		res = b - a * x
		#print(res)
	

		# until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		#norm_x = mynorm2<L>(x, dim);
		norm_x = mpmath.norm(x)
		#norm_res = mynorm2<L>(res, dim);
		norm_res = mpmath.norm(res)
		#print(itimes, norm_x, norm_res)

		if norm_res < mpmath.sqrt(mpmath.mpf(dim)) * rtol * norm_a * norm_x + atol:
			break

		# normalization: res := coef * res
		#normalization_coef = (L)1 / norm_res;
		normalization_coef = mpmath.mpmathify(1) / norm_res
		#myscal<L>(res, normalization_coef, res, dim);
		res = normalization_coef * res

		# Demote the residual from long precision to short precison
		#set_array(resf, res, dim);
		resf = get_qdvector_mpvec(resf, res)

		# Back-solve on short precision residual and short precision factors
		#solve_LU_linear_eq<S>(zf, af, resf, dim, af_ch);
		SolveQDLSPM(zf, af, resf, af_ch)
		#SolveQDLSP(zf, af, resf, af_ch)

		# Promote the correction from short precision to long precision
		#set_array(z, zf, dim);
		z = set_qdvector_mpvec(z, zf)

		# reverse normalizationi
		#myscal<L>(z, norm_res, z, dim);
		z = norm_res * z

		# Update solution in long precision
		#myaxpy<L>(x, (L)1, x, z, dim);
		#print(x, x+z)
		x = x + z
		#print(x)

		# for debug
		#std::cout << itimes << ", " << std::setprecision(10) << norm_res << std::endl;
		print(f'{itimes:4d}, {mpmath.nstr(norm_res, 10):s}, {mpmath.nstr(mpmath.norm(z), 10):s}')


	# if fail, retry in mpf_t precision
	if itimes >= maxtimes:
		# mpf_t precision
		#LU<L>(a, dim, af_ch);
		# SolveMPFLS(x, a, b);
		#solve_LU_linear_eq<L>(x, a, b, dim, af_ch);
		x = mpmath.lu_solve(a, b)

	#x = mpmath.lu_solve(a, b)

	# Clear
	free_qdmatrix(af)
	free_qdvector(bf)
	free_qdvector(xf)
	free_qdvector(resf)
	del res
	del zf
	del z
	del af_ch

	return x, itimes


# interative refinement with single & mpf_t mixed precision arithmetic
# solve x = a * b where known a in M_n(R) and b in R^n, unknown x in R^n
# unsigned long long_prec, short_prec; // long_prec > short_prec
def iterative_refinement_d_mp(x, a, b, maxtimes, rtol, atol): #rtol = 1.0e-30, atol = 0.0):

	#int i, itimes;
	#int *af_ch;
	dim = x.rows

	print('dim = ', dim)
	print('prec = ', mpmath.mp.prec)
	# Initialize
	af = mpmath.fp.matrix(dim, dim)
	bf = mpmath.fp.matrix([0.0 for i in range(dim)])
	xf = mpmath.fp.matrix([0.0 for i in range(dim)])

	res = init_mpvector(dim)

	resf = mpmath.fp.matrix([0.0 for i in range(dim)])
	z = init_mpvector(dim)
	zf = mpmath.fp.matrix([0.0 for i in range(dim)])
	#af_ch = (ct.c_long * (dim))
	#print('z.length = ', len(z))

	# norm_a := ||A||_F
	norm_a = mpmath.mnorm(a, 'fro')

	print('||A||_F = ', mpmath.nstr(norm_a, 10))
	# Make short precision copy of A and b
	#set_array(af, a, dim * dim);
	#get_ddmatrix_mpmat(af, a)
	get_fpmatrix_mpmat(af, a)
	#print('Convert A - A^(S)')
	#print(af)

	#set_array(bf, b, dim);
	get_fpvector_mpvec(bf, b)
	#print('Convert b - b^(S)')
	#print(bf)

	# Compute LU factorization in short precision
	# LU decomposition with partial pivoting
	#LU<S>(af, dim, af_ch);
	#DDLUdecompP(af, af_ch)
	xf = mpmath.lu_solve(af, bf)
	#print('LU ')

	# Apply back-solve in short precision with short precision factors
	#solve_LU_linear_eq<S>(xf, af, bf, dim, af_ch);
	#SolveDDLSP(xf, af, bf, af_ch)

	# Promote te solution from short precision to long precision
	#set_array(x, xf, dim);
	set_fpvector_mpvec(x, xf)

	print('iterative_ref start!')
	# repeat iterative refinement process
	for itimes in range(maxtimes):
		# Compute residual in long precision
		# res = b - a * x
		#mymv<L>(z, a, x, dim);
		#myaxpy<L>(res, (L)(-1), z, b, dim);
		#res = mpmath.residual(a, x, b)
		res = b - a * x
		#print(res)
	

		# until ||r_i||_2 < sqrt(n) * reps * ||A||_F * ||x_i||_2
		#norm_x = mynorm2<L>(x, dim);
		norm_x = mpmath.norm(x)
		#norm_res = mynorm2<L>(res, dim);
		norm_res = mpmath.norm(res)
		#print(itimes, norm_x, norm_res)

		if norm_res < mpmath.sqrt(mpmath.mpf(dim)) * rtol * norm_a * norm_x + atol:
			break

		# normalization: res := coef * res
		#normalization_coef = (L)1 / norm_res;
		normalization_coef = mpmath.mpmathify(1) / norm_res
		#myscal<L>(res, normalization_coef, res, dim);
		res = normalization_coef * res

		# Demote the residual from long precision to short precison
		#set_array(resf, res, dim);
		get_fpvector_mpvec(resf, res)
		
		# Back-solve on short precision residual and short precision factors
		#solve_LU_linear_eq<S>(zf, af, resf, dim, af_ch);
		#SolveDDLSP(zf, af, resf, af_ch)
		zf = mpmath.lu_solve(af, resf)

		# Promote the correction from short precision to long precision
		#set_array(z, zf, dim);
		set_fpvector_mpvec(z, zf)
	
		# reverse normalizationi
		#myscal<L>(z, norm_res, z, dim);
		z = norm_res * z

		# Update solution in long precision
		#myaxpy<L>(x, (L)1, x, z, dim);
		#print(x, x+z)
		x = x + z
		#print(x)

		# for debug
		#std::cout << itimes << ", " << std::setprecision(10) << norm_res << std::endl;
		print(f'{itimes:4d}, {mpmath.nstr(norm_res, 10):s}, {mpmath.nstr(mpmath.norm(z), 10):s}')


	# if fail, retry in mpf_t precision
	if itimes >= maxtimes:
		# mpf_t precision
		#LU<L>(a, dim, af_ch);
		# SolveMPFLS(x, a, b);
		#solve_LU_linear_eq<L>(x, a, b, dim, af_ch);
		x = mpmath.lu_solve(a, b)

	# Clear
	del af
	del bf
	del xf
	del resf
	del res
	del zf
	del z
	#del af_ch

	return itimes

# ---- end of BNCamPy ---

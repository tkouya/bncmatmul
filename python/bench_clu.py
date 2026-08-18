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
print('mpmath.prec = ', mpmath.mp.prec, 'mpmath.dps = ', mpmath.mp.dps)

mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq_complex(mid_dim)
mp_vec_x = [mpc_zero] * (mid_dim)
#print('A = ', mp_mat_a)
#print('b = ', mp_vec_b)
#print('x = ', mp_vec_true_x)

# Vector variables
#dd_vec_true_x_re = [dd_zero] * (mid_dim)
#dd_vec_true_x_im = [dd_zero] * (mid_dim)
cddfloat_vec_true_x = [cdd_zero] * (mid_dim) 
#ptr_dd_vec_true_x_re = (ct.c_double * (mid_dim * 2))()
#ptr_dd_vec_true_x_im = (ct.c_double * (mid_dim * 2))()

#dd_vec_b_re = [dd_zero] * (mid_dim)
#dd_vec_b_im = [dd_zero] * (mid_dim)
cddfloat_vec_b = [cdd_zero] * (mid_dim)
#ptr_dd_vec_b_re = (ct.c_double * (row_dim * 2))()
#ptr_dd_vec_b_im = (ct.c_double * (row_dim * 2))()

# set elements of vectors
cddfloat_vec_true_x = []
for i in range(mid_dim):
	#dd_vec_a[i] = rdd.mpfr_get_dd(mpfr_sqrt5 * (i + 1))
	cddfloat_vec_true_x.append(mpc_get_cdd_float(mp_vec_true_x[i]))
	#ptr_dd_vec_true_x[i * 2]     = dd_vec_true_x[i].val[0]
	#ptr_dd_vec_true_x[i * 2 + 1] = dd_vec_true_x[i].val[1]
	#print(cddfloat_vec_true_x[i])

#print(cddfloat_vec_true_x)

# cddvector
cddvec_true_x = init_cddvector(mid_dim)
#set_cddvector_cddfloat(cddvec_true_x, cddfloat_vec_true_x, mid_dim)
# https://stackoverflow.com/questions/55968037/expected-lp-c-float-instance-instead-of-list
#set_cddvector_cddfloat(cddvec_true_x, ((rcdd.cdd_float)*len(cddfloat_vec_true_x))(*cddfloat_vec_true_x), mid_dim)
set_cddvector_cdd_float(cddvec_true_x, cddfloat_vec_true_x)
#set_cddvector_ddvec(cddvec_true_x, dd_vec_true_x_re, dd_vec_true_x_im)

cddvec_b = init_cddvector(row_dim)
cddvec_x = init_cddvector(mid_dim)

# Matrix variables
cddfloat_mat_a = [cdd_zero] * (row_dim * mid_dim)
#ptr_dd_mat_a = (ct.c_double * (row_dim * mid_dim * 2))()

# set matrices
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j

		# set elements of A
		# A := sqrt(2) * [(i + j + 1)]
		#dd_mat_a[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt2 * (i + j + 1))
		cddfloat_mat_a[ij_index] = mpc_get_cdd_float(mp_mat_a[i,j])
		#ptr_dd_mat_a[(ij_index) * 2]     = dd_mat_a[ij_index].val[0]
		#ptr_dd_mat_a[(ij_index) * 2 + 1] = dd_mat_a[ij_index].val[1]
		#print(cddfloat_mat_a[ij_index])


# cddmatrix
cddmat_a = init_cddmatrix(row_dim, mid_dim)
#set_cddmatrix_cddfloat(cddmat_a, cddfloat_mat_a, row_dim * mid_dim)
#set_cddmatrix_cddfloat(cddmat_a, ((rcdd.cdd_float)*len(cddfloat_mat_a))(*cddfloat_mat_a), row_dim * mid_dim)
set_cddmatrix_cdd_float(cddmat_a, cddfloat_mat_a)
#print('cddmat_a = ')
#print_cddmatrix(cddmat_a)

# mat_ddmatrix_ddvec
start_time = time.time()
mul_cddmatrix_cddvec(cddvec_b, cddmat_a, cddvec_true_x)
end_time = time.time()
dd_mvmul_time = end_time - start_time
#print('cddvec_b = ')
#print_cddvector(cddvec_b)
#print('cddceb_b.re, im = ')
#print_ddvector(cddvec_b.contents.re)
#print_ddvector(cddvec_b.contents.im)
#set_cddfloat_cddvec((rcdd.cdd_float)(*cddfloat_vec_b), mid_dim, cddvec_b) #, row_dim)
#set_cddfloat_cddvec(((rcdd.cdd_float)*mid_dim)(*cddfloat_vec_b), mid_dim, cddvec_b) #, row_dim)
#set_cdd_float_cddvector(cddfloat_vec_b, cddvec_b) #, row_dim)
cddfloat_vec_b = set_cdd_float_cddvector(cddvec_b) #, row_dim)
#print(cddvec_b)
#for i in range(mid_dim):
#	print(i, '->', cddfloat_vec_b[i])
	#print(i, '->', cddfloat_vec_b[i].val_re[1], cddfloat_vec_b[i].val_im[1])

# LU 分解(mpmath)
if mpmath_skip_flag != True:
	start_time = time.time()
	mp_vec_x = mpmath.lu_solve(mp_mat_a, mp_vec_b)
	end_time = time.time()
	mp_lusolve_time = end_time - start_time
	#print('x = \n', mp_vec_x)
	max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
	print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))
	# print(mp_mat_a)

# CDDLU
ptr_pivot = (ct.c_long * (mid_dim))()
start_time = time.time()
ret_ddlu = CDDLUdecompPM(cddmat_a, ptr_pivot)
#ret_ddlu = CDDLUdecompP(cddmat_a, ptr_pivot)
#print_cddmatrix(cddmat_a)
pivot = [ptr_pivot[i] for i in range(mid_dim)]
#cddmatrix_lu = set_cdd_float_cddmatrix(cddmat_a)
#ret_ddlu = CDDLUdecompP(cddmat_a, ptr_pivot)
SolveCDDLSPM(cddvec_x, cddmat_a, cddvec_b, ptr_pivot)
#SolveCDDLSP(cddvec_x, cddmat_a, cddvec_b, ptr_pivot)
end_time = time.time()
dd_lusolve_time = end_time - start_time
#pivot = [ptr_pivot[i] for i in range(mid_dim)]
#print('ptr_pivot = ', pivot )
#print('ret of LU = ', ret_ddlu)
#print_cddvector(cddvec_x)
print('dd_LU = ', dd_lusolve_time)
if mpmath_skip_flag != True: print('mp_LU = ', mp_lusolve_time)

#for i in range(mid_dim):
#	print('x[', i, '] = ', ddvec_x.contents.element[0][i])
mp_vec_x = []
#mp_vec_true_x = []
for i in range(mid_dim):
	#mp_vec_x[i] = mpc_set_cdd_float(rcdd.cdd_float(
	#	cddvec_x.contents.re.contents.element[0][i],
	#	cddvec_x.contents.re.contents.element[1][i],
	#	cddvec_x.contents.im.contents.element[0][i],
	#	cddvec_x.contents.im.contents.element[1][i]
	#))
	mp_vec_x.append(mpc_set_cdd_float(rcdd.cdd_float(
		cddvec_x.contents.re.contents.element[0][i],
		cddvec_x.contents.re.contents.element[1][i],
		cddvec_x.contents.im.contents.element[0][i],
		cddvec_x.contents.im.contents.element[1][i]
	)))
	#print(cddvec_x.contents.re.contents.element[0][i], ' + ', cddvec_x.contents.im.contents.element[0][i], ' * I')
	#mp_vec_true_x.append(mpc_set_cdd_float(rcdd.cdd_float(
	#	cddvec_true_x.contents.re.contents.element[0][i], 
	#	cddvec_true_x.contents.re.contents.element[1][i],
	#	cddvec_true_x.contents.im.contents.element[0][i], 
	#	cddvec_true_x.contents.im.contents.element[1][i]
	#)))
cddfloat_vec_x = set_cdd_float_cddvector(cddvec_x) 
max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
#print(mp_vec_x)
#print(mp_vec_true_x)
print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# delete
del cddvec_true_x, cddfloat_vec_true_x, cddfloat_vec_b
free_cddvector(cddvec_x); free_cddvector(cddvec_b)
del mp_mat_a
del cddfloat_mat_a
free_cddmatrix(cddmat_a)

# ------
# TD
# ----- 
# mpfr
mpmath.mp.prec = 159 # TD
print('mpmath.prec = ', mpmath.mp.prec, 'mpmath.dps = ', mpmath.mp.dps)

mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq_complex(mid_dim)
#print('A = ', mp_mat_a)
#print('b = ', mp_vec_b)
#print('x = ', mp_vec_true_x)

# Vector variables
ctdfloat_vec_true_x = [ctd_zero] * (mid_dim) 
ctdfloat_vec_b = [ctd_zero] * (mid_dim)

# set elements of vectors
for i in range(mid_dim):
	ctdfloat_vec_true_x[i] = mpc_get_ctd_float(mp_vec_true_x[i])
	#print(ctdfloat_vec_true_x[i])

#print(ctdfloat_vec_true_x)

# ctdvector
ctdvec_true_x = init_ctdvector(mid_dim)
#set_cddvector_cddfloat(cddvec_true_x, cddfloat_vec_true_x, mid_dim)
set_ctdvector_ctd_float(ctdvec_true_x, ctdfloat_vec_true_x)
#set_cddvector_ddvec(cddvec_true_x, dd_vec_true_x_re, dd_vec_true_x_im)

ctdvec_b = init_ctdvector(row_dim)
ctdvec_x = init_ctdvector(mid_dim)

# Matrix variables
ctdfloat_mat_a = [ctd_zero] * (row_dim * mid_dim)
#ptr_dd_mat_a = (ct.c_double * (row_dim * mid_dim * 2))()

# set matrices
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j

		# set elements of A
		# A := sqrt(2) * [(i + j + 1)]
		#dd_mat_a[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt2 * (i + j + 1))
		ctdfloat_mat_a[ij_index] = mpc_get_ctd_float(mp_mat_a[i,j])


# ctdmatrix
ctdmat_a = init_ctdmatrix(row_dim, mid_dim)
#set_ctdmatrix_cddfloat(ctdmat_a, ctdfloat_mat_a, row_dim * mid_dim)
set_ctdmatrix_ctd_float(ctdmat_a, ctdfloat_mat_a)

# mat_tdmatrix_tdvec
start_time = time.time()
mul_ctdmatrix_ctdvec(ctdvec_b, ctdmat_a, ctdvec_true_x)
end_time = time.time()
td_mvmul_time = end_time - start_time

# LU 分解(mpmath)
if mpmath_skip_flag != True:
	start_time = time.time()
	mp_vec_x = mpmath.lu_solve(mp_mat_a, mp_vec_b)
	end_time = time.time()
	mp_lusolve_time = end_time - start_time
	#print('x = \n', mp_vec_x)
	max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
	print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# CTDLU
ptr_pivot = (ct.c_long * (mid_dim))()
start_time = time.time()
ret_ddlu = CTDLUdecompPM(ctdmat_a, ptr_pivot)
#ret_tdlu = CTDLUdecompP(ctdmat_a, ptr_pivot)
SolveCTDLSPM(ctdvec_x, ctdmat_a, ctdvec_b, ptr_pivot)
#SolveCTDLSP(ctdvec_x, ctdmat_a, ctdvec_b, ptr_pivot)
end_time = time.time()
td_lusolve_time = end_time - start_time
#print('ret of LU = ', ret_ddlu)
print('td_LU = ', td_lusolve_time)
if mpmath_skip_flag != True: print('mp_LU = ', mp_lusolve_time)

#for i in range(mid_dim):
#	print('x[', i, '] = ', ddvec_x.contents.element[0][i])
for i in range(mid_dim):
	mp_vec_x[i] = mpc_set_ctd_float(rcdd.ctd_float(
		ctdvec_x.contents.re.contents.element[0][i],
		ctdvec_x.contents.re.contents.element[1][i],
		ctdvec_x.contents.re.contents.element[2][i],
		ctdvec_x.contents.im.contents.element[0][i],
		ctdvec_x.contents.im.contents.element[1][i],
		ctdvec_x.contents.im.contents.element[2][i]
	))
	mp_vec_true_x[i] = mpc_set_ctd_float(rcdd.ctd_float(
		ctdvec_true_x.contents.re.contents.element[0][i], 
		ctdvec_true_x.contents.re.contents.element[1][i],
		ctdvec_true_x.contents.re.contents.element[2][i],
		ctdvec_true_x.contents.im.contents.element[0][i], 
		ctdvec_true_x.contents.im.contents.element[1][i],
		ctdvec_true_x.contents.im.contents.element[2][i]
	))

max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# delete
del ctdvec_true_x, ctdfloat_vec_true_x, ctdfloat_vec_b
free_ctdvector(ctdvec_x); free_ctdvector(ctdvec_b)
del mp_mat_a
del ctdfloat_mat_a
free_ctdmatrix(ctdmat_a)

# ------
# QD
# ----- 
# mpfr
mpmath.mp.prec = 212 # QD
print('mpmath.prec = ', mpmath.mp.prec, 'mpmath.dps = ', mpmath.mp.dps)

mp_mat_a, mp_vec_b, mp_vec_true_x, mp_mat_d = get_mptest_linear_eq_complex(mid_dim)
#mp_vec_x = mp_vec_true_x
#print('A = ', mp_mat_a)
#print('b = ', mp_vec_b)
#print('x = ', mp_vec_true_x)

# Vector variables
cqdfloat_vec_true_x = [cqd_zero] * (mid_dim) 
cqdfloat_vec_b = [cqd_zero] * (mid_dim)

# set elements of vectors
for i in range(mid_dim):
	cqdfloat_vec_true_x[i] = mpc_get_cqd_float(mp_vec_true_x[i])
	#print(cqdfloat_vec_true_x[i])

#print(cqdfloat_vec_true_x)

# cqdvector
cqdvec_true_x = init_cqdvector(mid_dim)
#set_cqdvector_cqdfloat(cqdvec_true_x, cqdfloat_vec_true_x, mid_dim)
set_cqdvector_cqd_float(cqdvec_true_x, cqdfloat_vec_true_x)

cqdvec_b = init_cqdvector(row_dim)
cqdvec_x = init_cqdvector(mid_dim)

# Matrix variables
cqdfloat_mat_a = [cqd_zero] * (row_dim * mid_dim)
#ptr_dd_mat_a = (ct.c_double * (row_dim * mid_dim * 2))()

# set matrices
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j

		# set elements of A
		# A := sqrt(2) * [(i + j + 1)]
		#qd_mat_a[ij_index] = rdd.mpfr_get_qd(mpfr_sqrt2 * (i + j + 1))
		cqdfloat_mat_a[ij_index] = mpc_get_cqd_float(mp_mat_a[i,j])


# cqdmatrix
cqdmat_a = init_cqdmatrix(row_dim, mid_dim)
#set_cqdmatrix_cqdfloat(cqdmat_a, cqdfloat_mat_a, row_dim * mid_dim)
set_cqdmatrix_cqd_float(cqdmat_a, cqdfloat_mat_a)

# mat_qdmatrix_qdvec
start_time = time.time()
mul_cqdmatrix_cqdvec(cqdvec_b, cqdmat_a, cqdvec_true_x)
end_time = time.time()
td_mvmul_time = end_time - start_time

# LU 分解(mpmath)
if mpmath_skip_flag != True:
	start_time = time.time()
	mp_vec_x = mpmath.lu_solve(mp_mat_a, mp_vec_b)
	end_time = time.time()
	mp_lusolve_time = end_time - start_time
	#print('x = \n', mp_vec_x)
	max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
	print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# CTDLU
ptr_pivot = (ct.c_long * (mid_dim))()
start_time = time.time()
ret_qdlu = CQDLUdecompPM(cqdmat_a, ptr_pivot)
#ret_qdlu = CQDLUdecompP(cqdmat_a, ptr_pivot)
SolveCQDLSPM(cqdvec_x, cqdmat_a, cqdvec_b, ptr_pivot)
#SolveCQDLSP(cqdvec_x, cqdmat_a, cqdvec_b, ptr_pivot)
end_time = time.time()
qd_lusolve_time = end_time - start_time
#print('ret of LU = ', ret_ddlu)
print('qd_LU = ', qd_lusolve_time)
if mpmath_skip_flag != True:print('mp_LU = ', mp_lusolve_time)

#for i in range(mid_dim):
#	print('x[', i, '] = ', ddvec_x.contents.element[0][i])
for i in range(mid_dim):
	mp_vec_x[i] = mpc_set_cqd_float(rcdd.cqd_float(
		cqdvec_x.contents.re.contents.element[0][i],
		cqdvec_x.contents.re.contents.element[1][i],
		cqdvec_x.contents.re.contents.element[2][i],
		cqdvec_x.contents.re.contents.element[3][i],
		cqdvec_x.contents.im.contents.element[0][i],
		cqdvec_x.contents.im.contents.element[1][i],
		cqdvec_x.contents.im.contents.element[2][i],
		cqdvec_x.contents.im.contents.element[3][i]
	))
	mp_vec_true_x[i] = mpc_set_cqd_float(rcdd.cqd_float(
		cqdvec_true_x.contents.re.contents.element[0][i], 
		cqdvec_true_x.contents.re.contents.element[1][i],
		cqdvec_true_x.contents.re.contents.element[2][i],
		cqdvec_true_x.contents.re.contents.element[3][i],
		cqdvec_true_x.contents.im.contents.element[0][i], 
		cqdvec_true_x.contents.im.contents.element[1][i],
		cqdvec_true_x.contents.im.contents.element[2][i],
		cqdvec_true_x.contents.im.contents.element[3][i],
	))

max_rel, min_rel, norm_rel = relerr(mp_vec_x, mp_vec_true_x)
print('max_rel, min_rel, norm_rel = ', mpmath.nstr(max_rel, 3), mpmath.nstr(min_rel, 3), mpmath.nstr(norm_rel, 3))

# delete
del cqdvec_true_x, cqdfloat_vec_true_x, cqdfloat_vec_b
free_cqdvector(cqdvec_x); free_cqdvector(cqdvec_b)
del mp_mat_a
del cqdfloat_mat_a
free_cqdmatrix(cqdmat_a)


# end
librdd.rdd_end()

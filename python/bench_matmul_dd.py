# bench_matmul_dd.py: LIBRDD test

import ctypes as ct
import rdd
import gmpy2
import time

# gmpy2
gmpy2.get_context().precision = 212 # in bits
print(gmpy2.get_context())

mpfr_sqrt2 = gmpy2.sqrt(gmpy2.mpfr(2))
mpfr_sqrt3 = gmpy2.sqrt(gmpy2.mpfr(3))

# RDD library
librdd = ct.cdll.LoadLibrary('../librdd.so');

# start
librdd.rdd_start();

# 自作行列乗算
def xd_mymatmul(mat_a, row_dim_mat_a, col_dim_mat_a, mat_b, row_dim_mat_b, col_dim_mat_b, xd_zero):
	row_dim  , mid_dim = row_dim_mat_a, col_dim_mat_a
	mid_dim_b, col_dim = row_dim_mat_b, col_dim_mat_b

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

# 自作行列乗算
for sq_dim in [32, 64, 128, 256]:

	# dimension
	row_dim = sq_dim
	mid_dim = sq_dim
	col_dim = sq_dim

	dd_zero = rdd.dd_float(0.0, 0.0)

	# DD行列データ型
	# A := sqrt(2) * [(i + j + 1)]
	dd_mat_a = [dd_zero] * (row_dim * mid_dim)

	ptr_dd_mat_a = (ct.c_double * (row_dim * mid_dim * 2))()

	for i in range(row_dim):
		for j in range(mid_dim):
			ij_index = i * mid_dim + j
			dd_mat_a[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt2 * (i + j + 1))

			ptr_dd_mat_a[(ij_index) * 2]     = dd_mat_a[ij_index].val[0]
			ptr_dd_mat_a[(ij_index) * 2 + 1] = dd_mat_a[ij_index].val[1]

	# B := sqrt(3) * [max(i + 1, j + 1)]
	dd_mat_b = [dd_zero] * (mid_dim * col_dim)

	ptr_dd_mat_b = (ct.c_double * (mid_dim * col_dim * 2))()
	ptr_dd_mat_c = (ct.c_double * (row_dim * col_dim * 2))()

	for i in range(mid_dim):
		for j in range(col_dim):
			ij_index = i * col_dim + j
			if i > j:
				dd_mat_b[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt3 * (i + 1))
			else:
				dd_mat_b[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt3 * (j + 1))

			ptr_dd_mat_b[(ij_index) * 2]     = dd_mat_b[ij_index].val[0]
			ptr_dd_mat_b[(ij_index) * 2 + 1] = dd_mat_b[ij_index].val[1]

#	print('dd_mat_a, b[0, 0]      = ', dd_mat_a[0], dd_mat_b[0])
#	print('ptr_dd_mat_a, b[0, 0]  = ', rdd.dd_float(ptr_dd_mat_a[0], ptr_dd_mat_a[1]), rdd.dd_float(ptr_dd_mat_b[0], ptr_dd_mat_b[1]))

	# dd
	start_time = time.time()
	dd_mat_c = xd_mymatmul(dd_mat_a, row_dim, mid_dim, dd_mat_b, mid_dim, col_dim, dd_zero)
	end_time = time.time()
	dd_matmul_time = end_time - start_time

	# ptr_dd
	start_time = time.time()
	rdd.dd_matmul_simple(ptr_dd_mat_c, row_dim, col_dim, ptr_dd_mat_a, row_dim, mid_dim, ptr_dd_mat_b, mid_dim, col_dim)
	end_time = time.time()
	ptr_dd_matmul_time = end_time - start_time

	print('dim = ', sq_dim, f',     dd: {dd_matmul_time:5.3f}')
	print('dim = ', sq_dim, f', ptr_dd: {ptr_dd_matmul_time:5.3f}')

#	print('dd_mat_c[0, 0]    = ', dd_mat_c[0])
#	print('ptr_dd_mat_c[0, 0]= ', rdd.dd_float(ptr_dd_mat_c[0], ptr_dd_mat_c[1]))

	# delete
	del dd_mat_a, dd_mat_b, dd_mat_c
	del ptr_dd_mat_a, ptr_dd_mat_b, ptr_dd_mat_c

librdd.rdd_end();

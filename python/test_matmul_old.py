# test_matmul.py: Test script for MV and MM

from bncampy import *

# start
librdd.rdd_start();

sq_dim = 32

# dd_float
row_dim = sq_dim
mid_dim = sq_dim
col_dim = sq_dim
dd_zero = rdd.dd_float(0.0, 0.0)

# A := sqrt(2) * [(i + j + 1)]
dd_mat_a = [dd_zero] * (row_dim * mid_dim)
print('size_mat_a = ', len(dd_mat_a))
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j
		dd_mat_a[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt2 * (i + j + 1))

# B := sqrt(3) * [max(i + 1, j + 1)]
dd_mat_b = [dd_zero] * (mid_dim * col_dim)
print('size_mat_b = ', len(dd_mat_b))
for i in range(mid_dim):
	for j in range(col_dim):
		ij_index = i * col_dim + j
		if i > j:
			dd_mat_b[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt3 * (i + 1))
		else:
			dd_mat_b[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt3 * (j + 1))

# 自作行列乗算
def dd_mymatmul(mat_a, row_dim_mat_a, col_dim_mat_a, mat_b, row_dim_mat_b, col_dim_mat_b):
	row_dim  , mid_dim = row_dim_mat_a, col_dim_mat_a
	mid_dim_b, col_dim = row_dim_mat_b, col_dim_mat_b
 #   print('row_dim, mid_dim, col_dim = ', row_dim, mid_dim, col_dim)

	zero = rdd.dd_float(0.0, 0.0)

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

start_time = time.time()
dd_mat_c = dd_mymatmul(dd_mat_a, row_dim, mid_dim, dd_mat_b, mid_dim, col_dim)
end_time = time.time()
print('dd_mymatmul 計算時間(秒): ', end_time - start_time)

# td_float
row_dim = sq_dim
mid_dim = sq_dim
col_dim = sq_dim
td_zero = rdd.td_float(0.0, 0.0, 0.0)

# A := sqrt(2) * [(i + j + 1)]
td_mat_a = [td_zero] * (row_dim * mid_dim)
print('size_mat_a = ', len(td_mat_a))
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j
		td_mat_a[ij_index] = rdd.mpfr_get_td(mpfr_sqrt2 * (i + j + 1))

# B := sqrt(3) * [max(i + 1, j + 1)]
td_mat_b = [td_zero] * (mid_dim * col_dim)
print('size_mat_b = ', len(td_mat_b))
for i in range(mid_dim):
	for j in range(col_dim):
		ij_index = i * col_dim + j
		if i > j:
			td_mat_b[ij_index] = rdd.mpfr_get_td(mpfr_sqrt3 * (i + 1))
		else:
			td_mat_b[ij_index] = rdd.mpfr_get_td(mpfr_sqrt3 * (j + 1))

# 自作行列乗算
def td_mymatmul(mat_a, row_dim_mat_a, col_dim_mat_a, mat_b, row_dim_mat_b, col_dim_mat_b):
	row_dim  , mid_dim = row_dim_mat_a, col_dim_mat_a
	mid_dim_b, col_dim = row_dim_mat_b, col_dim_mat_b
 #   print('row_dim, mid_dim, col_dim = ', row_dim, mid_dim, col_dim)

	zero = rdd.td_float(0.0, 0.0, 0.0)

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
start_time = time.time()
td_mat_c = td_mymatmul(td_mat_a, row_dim, mid_dim, td_mat_b, mid_dim, col_dim)
end_time = time.time()
print('td_mymatmul 計算時間(秒): ', end_time - start_time)


# qd_float
row_dim = sq_dim
mid_dim = sq_dim
col_dim = sq_dim
qd_zero = rdd.qd_float(0.0, 0.0, 0.0, 0.0)

# A := sqrt(2) * [(i + j + 1)]
qd_mat_a = [qd_zero] * (row_dim * mid_dim)
print('size_mat_a = ', len(qd_mat_a))
for i in range(row_dim):
	for j in range(mid_dim):
		ij_index = i * mid_dim + j
		qd_mat_a[ij_index] = rdd.mpfr_get_qd(mpfr_sqrt2 * (i + j + 1))

# B := sqrt(3) * [max(i + 1, j + 1)]
qd_mat_b = [qd_zero] * (mid_dim * col_dim)
print('size_mat_b = ', len(qd_mat_b))
for i in range(mid_dim):
	for j in range(col_dim):
		ij_index = i * col_dim + j
		if i > j:
			qd_mat_b[ij_index] = rdd.mpfr_get_qd(mpfr_sqrt3 * (i + 1))
		else:
			qd_mat_b[ij_index] = rdd.mpfr_get_qd(mpfr_sqrt3 * (j + 1))

# 自作行列乗算
def qd_mymatmul(mat_a, row_dim_mat_a, col_dim_mat_a, mat_b, row_dim_mat_b, col_dim_mat_b):
	row_dim  , mid_dim = row_dim_mat_a, col_dim_mat_a
	mid_dim_b, col_dim = row_dim_mat_b, col_dim_mat_b
 #   print('row_dim, mid_dim, col_dim = ', row_dim, mid_dim, col_dim)

	zero = rdd.qd_float(0.0, 0.0, 0.0, 0.0)

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
start_time = time.time()
qd_mat_c = qd_mymatmul(qd_mat_a, row_dim, mid_dim, qd_mat_b, mid_dim, col_dim)
end_time = time.time()
print('qd_mymatmul 計算時間(秒): ', end_time - start_time)


librdd.rdd_end();

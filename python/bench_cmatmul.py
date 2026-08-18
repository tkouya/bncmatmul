# load BNCamPy
from bncampy import *

# start
librdd.rdd_start();

mpfr_sqrt2 = gmpy2.sqrt(gmpy2.mpfr('2'))
mpfr_sqrt3 = gmpy2.sqrt(gmpy2.mpfr(3))
mpfr_sqrt5 = gmpy2.sqrt(gmpy2.mpfr(5))
mp_sqrt2 = mpmath.sqrt(mpmath.mp.mpf('2'))
mp_sqrt3 = mpmath.sqrt(mpmath.mp.mpf(3))
mp_sqrt5 = mpmath.sqrt(mpmath.mp.mpf(5))

#for sq_dim in [32, 64, 128, 256]:
#for sq_dim in [128, 256, 512, 1024]:
for sq_dim in [100, 200, 500, 1000]:

	# float

	# dimension
	row_dim = sq_dim
	mid_dim = sq_dim
	col_dim = sq_dim

	# ベクトルデータ型
	d_vec_a  = [ d_zero] * (mid_dim)
	dd_vec_a = [dd_zero] * (mid_dim)
	td_vec_a = [td_zero] * (mid_dim)
	qd_vec_a = [qd_zero] * (mid_dim)
	mp_vec_a = [mp_zero] * (mid_dim)
	mpfr_vec_a = [mpfr_zero] * (mid_dim)
	cdd_vec_a = [cdd_zero] * (mid_dim)
	ctd_vec_a = [ctd_zero] * (mid_dim)
	cqd_vec_a = [cqd_zero] * (mid_dim)
	mpc_vec_a = [mpc_zero] * (mid_dim)

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
		cdd_vec_a[i] = mpc_get_cdd_float(mpmath.mpc(mp_sqrt5 * (i + 1), mp_sqrt2 * (i + 1)))
		ctd_vec_a[i] = mpc_get_ctd_float(mpmath.mpc(mp_sqrt5 * (i + 1), mp_sqrt2 * (i + 1)))
		cqd_vec_a[i] = mpc_get_cqd_float(mpmath.mpc(mp_sqrt5 * (i + 1), mp_sqrt2 * (i + 1)))
		mpc_vec_a[i] = mpmath.mp.mpc(mp_sqrt5 * mpmath.mp.mpf(i + 1), mp_sqrt2 * mpmath.mp.mpf(i + 1))


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
	cdd_mat_a = [cdd_zero] * (row_dim * mid_dim)
	ctd_mat_a = [ctd_zero] * (row_dim * mid_dim)
	cqd_mat_a = [cqd_zero] * (row_dim * mid_dim)
	mpc_mat_a = [mpc_zero] * (row_dim * mid_dim)


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
			cdd_mat_a[i] = mpc_get_cdd_float(mpmath.mpc(mp_sqrt5 * (i + j + 1), mp_sqrt2 * (i + j + 1)))
			ctd_mat_a[i] = mpc_get_ctd_float(mpmath.mpc(mp_sqrt5 * (i + j + 1), mp_sqrt2 * (i + j + 1)))
			cqd_mat_a[i] = mpc_get_cqd_float(mpmath.mpc(mp_sqrt5 * (i + j + 1), mp_sqrt2 * (i + j + 1)))
			mpc_mat_a[i] = mpmath.mp.mpc(mp_sqrt5 * mpmath.mp.mpf(i + j + 1), mp_sqrt2 * mpmath.mp.mpf(i + j + 1))
			
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
	cdd_mat_b = [cdd_zero] * (row_dim * mid_dim)
	ctd_mat_b = [ctd_zero] * (row_dim * mid_dim)
	cqd_mat_b = [cqd_zero] * (row_dim * mid_dim)
	mpc_mat_b = [mpc_zero] * (row_dim * mid_dim)

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
				cdd_mat_b[ij_index] = mpc_get_cdd_float(mpmath.mpc(mp_sqrt3 * (i + 1), mp_sqrt2 * (i + 1)))
				ctd_mat_b[ij_index] = mpc_get_ctd_float(mpmath.mpc(mp_sqrt3 * (i + 1), mp_sqrt2 * (i + 1)))
				cqd_mat_b[ij_index] = mpc_get_cqd_float(mpmath.mpc(mp_sqrt3 * (i + 1), mp_sqrt2 * (i + 1)))
				mpc_mat_b[ij_index] = mpmath.mp.mpc(mp_sqrt3 * mpmath.mp.mpf(i + 1), mp_sqrt2 * mpmath.mp.mpf(i + 1))
			else:
				d_mat_b[ij_index]  = float(mpfr_sqrt3 * (j + 1))
				dd_mat_b[ij_index] = rdd.mpfr_get_dd(mpfr_sqrt3 * (j + 1))
				td_mat_b[ij_index] = rdd.mpfr_get_td(mpfr_sqrt3 * (j + 1))
				qd_mat_b[ij_index] = rdd.mpfr_get_qd(mpfr_sqrt3 * (j + 1))
				mp_mat_b[ij_index] = mp_sqrt3 * mpmath.mp.mpf(j + 1)
				mpfr_mat_b[ij_index] = mpfr_sqrt3 * gmpy2.mpfr(j + 1)
				cdd_mat_b[ij_index] = mpc_get_cdd_float(mpmath.mpc(mp_sqrt3 * (j + 1), mp_sqrt2 * (j + 1)))
				ctd_mat_b[ij_index] = mpc_get_ctd_float(mpmath.mpc(mp_sqrt3 * (j + 1), mp_sqrt2 * (j + 1)))
				cqd_mat_b[ij_index] = mpc_get_cqd_float(mpmath.mpc(mp_sqrt3 * (j + 1), mp_sqrt2 * (j + 1)))
				mpc_mat_b[ij_index] = mpmath.mp.mpc(mp_sqrt3 * mpmath.mp.mpf(j + 1), mp_sqrt2 * mpmath.mp.mpf(j + 1))

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

	# cddvector
	cddvec_a = init_cddvector(mid_dim)
	set_cddvector_cdd_float(cddvec_a, cdd_vec_a)
	cddvec_b = init_cddvector(row_dim)

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

	# cddmatrix
	cddmat_a = init_cddmatrix(row_dim, mid_dim)
	set_cddmatrix_cdd_float(cddmat_a, cdd_mat_a)
	cddmat_b = init_cddmatrix(mid_dim, col_dim)
	set_cddmatrix_cdd_float(cddmat_b, cdd_mat_b)
	cddmat_c = init_cddmatrix(row_dim, col_dim)
	#print(cddmat_a)
	#print('size a = ', cddmat_a.contents.row_dim, cddmat_a.contents.col_dim)
	#print('size b = ', cddmat_b.contents.row_dim, cddmat_b.contents.col_dim)
	#print('size c = ', cddmat_c.contents.row_dim, cddmat_c.contents.col_dim)

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

    # cdd
	start_time = time.time()
	cdd_mat_c = xd_mymatmul(cdd_mat_a, row_dim, mid_dim, cdd_mat_b, mid_dim, col_dim, cdd_zero)
	end_time = time.time()
	cdd_matmul_time = end_time - start_time
	print('cdd(', 53 * 2, ')')
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

	# mat_cddmatrix_block
	start_time = time.time()
	mul_cddmatrix_block(cddmat_c, cddmat_a, cddmat_b, 32)
	end_time = time.time()
	cdd_matmul_block_time = end_time - start_time

	# mat_cddmatrix_strassen
	start_time = time.time()
	mul_cddmatrix_strassen(cddmat_c, cddmat_a, cddmat_b, 32)
	end_time = time.time()
	cdd_matmul_strassen_time = end_time - start_time

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

	# cdd_mv
	start_time = time.time()
	cdd_vec_b = xd_mymv(cdd_mat_a, row_dim, mid_dim, cdd_vec_a, mid_dim, cdd_zero)
	end_time = time.time()
	cdd_mymv_time = end_time - start_time

	# mat_ddmatrix_ddvec
	start_time = time.time()
	mul_ddmatrix_ddvec(ddvec_b, ddmat_a, ddvec_a)
	end_time = time.time()
	dd_mvmul_time = end_time - start_time

	# mat_cddmatrix_cddvec
	start_time = time.time()
	mul_cddmatrix_cddvec(cddvec_b, cddmat_a, cddvec_a)
	end_time = time.time()
	cdd_mvmul_time = end_time - start_time

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

	# ctdvector
	ctdvec_a = init_ctdvector(mid_dim)
	set_ctdvector_ctd_float(ctdvec_a, ctd_vec_a)
	ctdvec_b = init_ctdvector(row_dim)

	# ctdmatrix
	ctdmat_a = init_ctdmatrix(row_dim, mid_dim)
	set_ctdmatrix_ctd_float(ctdmat_a, ctd_mat_a)
	ctdmat_b = init_ctdmatrix(mid_dim, col_dim)
	set_ctdmatrix_ctd_float(ctdmat_b, ctd_mat_b)
	ctdmat_c = init_ctdmatrix(row_dim, col_dim)

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

	# ctd
	start_time = time.time()
	ctd_mat_c = xd_mymatmul(ctd_mat_a, row_dim, mid_dim, ctd_mat_b, mid_dim, col_dim, ctd_zero)
	end_time = time.time()
	ctd_matmul_time = end_time - start_time
	print('ctd(', 53 * 3, ')')

	# mat_ctdmatrix_block
	start_time = time.time()
	mul_ctdmatrix_block(ctdmat_c, ctdmat_a, ctdmat_b, 32)
	end_time = time.time()
	ctd_matmul_block_time = end_time - start_time
#	print('sq_dim = ', sq_dim, ' dd_mymatmul 計算時間(秒): ', end_time - start_time)

	# mat_ctdmatrix_strassen
	start_time = time.time()
	mul_ctdmatrix_strassen(ctdmat_c, ctdmat_a, ctdmat_b, 32)
	end_time = time.time()
	ctd_matmul_strassen_time = end_time - start_time

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

	# ctd_mv
	start_time = time.time()
	ctd_vec_b = xd_mymv(ctd_mat_a, row_dim, mid_dim, ctd_vec_a, mid_dim, ctd_zero)
	end_time = time.time()
	ctd_mymv_time = end_time - start_time

	# mat_tdmatrix_tdvec
	start_time = time.time()
	mul_ctdmatrix_ctdvec(ctdvec_b, ctdmat_a, ctdvec_a)
	end_time = time.time()
	ctd_mvmul_time = end_time - start_time
	
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

	# cqdvector
	cqdvec_a = init_cqdvector(mid_dim)
	set_cqdvector_cqd_float(cqdvec_a, cqd_vec_a)
	cqdvec_b = init_cqdvector(row_dim)

	# cqdmatrix
	cqdmat_a = init_cqdmatrix(row_dim, mid_dim)
	set_cqdmatrix_cqd_float(cqdmat_a, cqd_mat_a)
	cqdmat_b = init_cqdmatrix(mid_dim, col_dim)
	set_cqdmatrix_cqd_float(cqdmat_b, cqd_mat_b)
	cqdmat_c = init_cqdmatrix(row_dim, col_dim)

	# qd

	# qd_mv
	start_time = time.time()
	qd_vec_b = xd_mymv(qd_mat_a, row_dim, mid_dim, qd_vec_a, mid_dim, qd_zero)
	end_time = time.time()
	qd_mymv_time = end_time - start_time
	
	start_time = time.time()
	qd_mat_c = xd_mymatmul(qd_mat_a, row_dim, mid_dim, qd_mat_b, mid_dim, col_dim, qd_zero)
	end_time = time.time()
	qd_matmul_time = end_time - start_time
	print('qd(', 53 * 4, ')')
	#cProfile.run('xd_mymatmul(qd_mat_a, row_dim, mid_dim, qd_mat_b, mid_dim, col_dim, qd_zero)')
#	print('sq_dim = ', sq_dim, ' qd_mymatmul 計算時間(秒): ', end_time - start_time)

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

	# cqd
	start_time = time.time()
	cqd_mat_c = xd_mymatmul(cqd_mat_a, row_dim, mid_dim, cqd_mat_b, mid_dim, col_dim, cqd_zero)
	end_time = time.time()
	cqd_matmul_time = end_time - start_time
	print('cqd(', 53 * 4, ')')

	# mat_cqdmatrix_block
	start_time = time.time()
	mul_cqdmatrix_block(cqdmat_c, cqdmat_a, cqdmat_b, 32)
	end_time = time.time()
	cqd_matmul_block_time = end_time - start_time
#	print('sq_dim = ', sq_dim, ' dd_mymatmul 計算時間(秒): ', end_time - start_time)

	# mat_cqdmatrix_strassen
	start_time = time.time()
	mul_cqdmatrix_strassen(cqdmat_c, cqdmat_a, cqdmat_b, 32)
	end_time = time.time()
	cqd_matmul_strassen_time = end_time - start_time

	# cqd_mv
	start_time = time.time()
	cqd_vec_b = xd_mymv(cqd_mat_a, row_dim, mid_dim, cqd_vec_a, mid_dim, cqd_zero)
	end_time = time.time()
	cqd_mymv_time = end_time - start_time

	# mat_cqdmatrix_cqdvec
	start_time = time.time()
	mul_cqdmatrix_cqdvec(cqdvec_b, cqdmat_a, cqdvec_a)
	end_time = time.time()
	cqd_mvmul_time = end_time - start_time

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
	print('sq_dim = ', sq_dim, f', cdd, ctd, cqd: {cdd_matmul_time:5.3f}, {ctd_matmul_time:5.3f}, {cqd_matmul_time:5.3f}')

	print('sq_dim = ', sq_dim, f', ddmat , tdmat , qdmat(sec) : {dd_matmul_block_time:5.3f}, {td_matmul_block_time:5.3f}, {qd_matmul_block_time:5.3f}')
	print('sq_dim = ', sq_dim, f', ddmats, tdmats, qdmats(sec): {dd_matmul_strassen_time:5.3f}, {td_matmul_strassen_time:5.3f}, {qd_matmul_strassen_time:5.3f}')
	print('sq_dim = ', sq_dim, f', cddmat , ctdmat , cqdmat(sec) : {cdd_matmul_block_time:5.3f}, {ctd_matmul_block_time:5.3f}, {cqd_matmul_block_time:5.3f}')
	print('sq_dim = ', sq_dim, f', cddmats, ctdmats, cqdmats(sec): {cdd_matmul_strassen_time:5.3f}, {ctd_matmul_strassen_time:5.3f}, {cqd_matmul_strassen_time:5.3f}')
	print('sq_dim = ', sq_dim, f',   myddvec,   mytdvec,   myqdvec(sec): {dd_mymv_time:5.3g}, {td_mymv_time:5.3g}, {qd_mymv_time:5.3g}')
	print('sq_dim = ', sq_dim, f',   mycddvec,   myctdvec,   mycqdvec(sec): {cdd_mymv_time:5.3g}, {ctd_mymv_time:5.3g}, {cqd_mymv_time:5.3g}')
	print('sq_dim = ', sq_dim, f', ptr_ddvec, ptr_tdvec, ptr_qdvec(sec): {ptr_dd_mvmul_time:5.3g}, {ptr_td_mvmul_time:5.3g}, {ptr_qd_mvmul_time:5.3g}')
	print('sq_dim = ', sq_dim, f',     ddvec,     tdvec,     qdvec(sec): {dd_mvmul_time:5.3g}, {td_mvmul_time:5.3g}, {qd_mvmul_time:5.3g}')
	print('sq_dim = ', sq_dim, f',     cddvec,     ctdvec,     cqdvec(sec): {cdd_mvmul_time:5.3g}, {ctd_mvmul_time:5.3g}, {cqd_mvmul_time:5.3g}')


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

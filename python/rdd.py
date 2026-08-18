# test_rdd.py: LIBRDD test

# ref
# 1. https://qiita.com/kuboshu83/items/e76d5fdeac6132734a07
# 2. https://docs.python.org/ja/3/library/ctypes.html

import ctypes as ct
import gmpy2

# MPFR library
libmpfr = ct.cdll.LoadLibrary('libmpfr.so')

# BNCmatmul library
#libbncmatmul = ct.cdll.LoadLibrary('libbncmatmul-0.23.so');

# RDD library
librdd = ct.cdll.LoadLibrary('librdd.so');

# mpfr_get_dd
def mpfr_get_dd(mpfr_val):
	dd_ret = dd_float()
	dd_ret.val[0] = float(mpfr_val)
	dd_ret.val[1] = float(mpfr_val - gmpy2.mpfr(dd_ret.val[0]))
	return dd_ret

# ret := mat_a * vec_b
#void dd_mvmul_simple(double *ret, int ret_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_dim)
def dd_mvmul_simple(vec_c, c_dim, mat_a, a_row_dim, a_col_dim, vec_b, b_dim):

	librdd.dd_mvmul_simple(
		vec_c, c_dim,
		mat_a, a_row_dim, a_col_dim,
		vec_b, b_dim
	)

#void dd_matmul_simple(double *ret, int ret_row_dim, int ret_col_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_row_dim, int b_col_dim)
#def dd_matmul_simple(mat_a, a_row_dim, a_col_dim, mat_b, b_row_dim, b_col_dim):
def dd_matmul_simple(mat_c, c_row_dim, c_col_dim, mat_a, a_row_dim, a_col_dim, mat_b, b_row_dim, b_col_dim):

#	ret = librdd.dd_mat_init(a_row_dim, b_col_dim)

	librdd.dd_matmul_simple(
		mat_c, c_row_dim, c_col_dim,
		mat_a, a_row_dim, a_col_dim,
		mat_b, b_row_dim, b_col_dim
	)

#	return ret

# ret := mat_a * vec_b
#void td_mvmul_simple(double *ret, int ret_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_dim)
def td_mvmul_simple(vec_c, c_dim, mat_a, a_row_dim, a_col_dim, vec_b, b_dim):

	librdd.td_mvmul_simple(
		vec_c, c_dim,
		mat_a, a_row_dim, a_col_dim,
		vec_b, b_dim
	)


#void td_matmul_simple(double *ret, int ret_row_dim, int ret_col_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_row_dim, int b_col_dim)
#def td_matmul_simple(mat_a, a_row_dim, a_col_dim, mat_b, b_row_dim, b_col_dim):
def td_matmul_simple(mat_c, c_row_dim, c_col_dim, mat_a, a_row_dim, a_col_dim, mat_b, b_row_dim, b_col_dim):

#	ret = librdd.td_mat_init(a_row_dim, b_col_dim)

	librdd.td_matmul_simple(
		mat_c, c_row_dim, c_col_dim,
		mat_a, a_row_dim, a_col_dim,
		mat_b, b_row_dim, b_col_dim
	)

#	return ret

# ret := mat_a * vec_b
#void qd_mvmul_simple(double *ret, int ret_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_dim)
def qd_mvmul_simple(vec_c, c_dim, mat_a, a_row_dim, a_col_dim, vec_b, b_dim):

	librdd.qd_mvmul_simple(
		vec_c, c_dim,
		mat_a, a_row_dim, a_col_dim,
		vec_b, b_dim
	)

#void qd_matmul_simple(double *ret, int ret_row_dim, int ret_col_dim, double *a, int a_row_dim, int a_col_dim, double *b, int b_row_dim, int b_col_dim)
#def qd_matmul_simple(mat_a, a_row_dim, a_col_dim, mat_b, b_row_dim, b_col_dim):
def qd_matmul_simple(mat_c, c_row_dim, c_col_dim, mat_a, a_row_dim, a_col_dim, mat_b, b_row_dim, b_col_dim):

#	ret = librdd.qd_mat_init(a_row_dim, b_col_dim)

	librdd.qd_matmul_simple(
		mat_c, c_row_dim, c_col_dim,
		mat_a, a_row_dim, a_col_dim,
		mat_b, b_row_dim, b_col_dim
	)

#	return ret

# mpfr_get_td
def mpfr_get_td(mpfr_val):
	td_ret = td_float()
	td_ret.val[0] = float(mpfr_val)
	td_ret.val[1] = float(mpfr_val - gmpy2.mpfr(td_ret.val[0]))
	td_ret.val[2] = float(mpfr_val - gmpy2.mpfr(td_ret.val[0]) - gmpy2.mpfr(td_ret.val[1]))
	return td_ret

# mpfr_get_td
def mpfr_get_qd(mpfr_val):
	qd_ret = qd_float()
	qd_ret.val[0] = float(mpfr_val)
	qd_ret.val[1] = float(mpfr_val - gmpy2.mpfr(qd_ret.val[0]))
	qd_ret.val[2] = float(mpfr_val - gmpy2.mpfr(qd_ret.val[0]) - gmpy2.mpfr(qd_ret.val[1]))
	qd_ret.val[3] = float(mpfr_val - gmpy2.mpfr(qd_ret.val[0]) - gmpy2.mpfr(qd_ret.val[1]) - gmpy2.mpfr(qd_ret.val[2]))
	return qd_ret

# dd_float
class dd_float(ct.Structure):
	_fields_ = [('val', ct.c_double * 2)]

	# constructor
	def __init__(self, x0 = None, x1 = None):
		if x0 != None: self.val[0] = x0
		else: self.val[0] = 0.0
		if x1 != None: self.val[1] = x1
		else: self.val[1] = 0.0

	# mpfr_set_dd
	def mpfr_set_dd(self, prec = None):
		old_prec = gmpy2.get_context().precision
		if prec != None:
			gmpy2.get_context().precision = prec
		r = gmpy2.mpfr('0.0')
#		with gmpy2.ieee(64) as ctx:
		mpfr_val = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val[0] = gmpy2.mpfr(self.val[0])
		mpfr_val[1] = gmpy2.mpfr(self.val[1])

		r = mpfr_val[0] + mpfr_val[1]

		gmpy2.get_context().precision = old_prec

		return r

	# print
	def __str__(self):
		tmp = self.mpfr_set_dd(53 * 2)
		return str(tmp)

	# addition
	def __add__(self, y):
		ret = dd_float()
		librdd.rdd_add(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# subtraction
	def __sub__(self, y):
		ret = dd_float()
		librdd.rdd_sub(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# multiplication
	def __mul__(self, y):
		ret = dd_float()
		librdd.rdd_mul(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# division
	def __truediv__(self, y):
		ret = dd_float()
		librdd.rdd_div(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

# td_float
class td_float(ct.Structure):
	_fields_ = [("val", ct.c_double * 3)]

	# constructor
	def __init__(self, x0 = None, x1 = None, x2 = None):
		if x0 != None: self.val[0] = x0
		else: self.val[0] = 0.0
		if x1 != None: self.val[1] = x1
		else: self.val[1] = 0.0
		if x2 != None: self.val[2] = x2
		else: self.val[2] = 0.0

	# mpfr_set_td
	def mpfr_set_td(self, prec = None):
		old_prec = gmpy2.get_context().precision
		if prec != None:
			gmpy2.get_context().precision = prec
		r = gmpy2.mpfr('0.0')
#		with gmpy2.ieee(64) as ctx:
		mpfr_val = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val[0] = gmpy2.mpfr(self.val[0])
		mpfr_val[1] = gmpy2.mpfr(self.val[1])
		mpfr_val[2] = gmpy2.mpfr(self.val[2])

		r = mpfr_val[0] + mpfr_val[1] + mpfr_val[2]

		gmpy2.get_context().precision = old_prec

		return r

	# print
	def __str__(self):
		tmp = self.mpfr_set_td(53 * 3)
		return str(tmp)

	# addition
	def __add__(self, y):
		ret = td_float()
		librdd.rtd_add(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# subtraction
	def __sub__(self, y):
		ret = td_float()
		librdd.rtd_sub(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# multiplication
	def __mul__(self, y):
		ret = td_float()
		librdd.rtd_mul(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# division
	def __truediv__(self, y):
		ret = td_float()
		librdd.rtd_div(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

# qd_float
class qd_float(ct.Structure):
	_fields_ = [("val", ct.c_double * 4)]

	# constructor
	def __init__(self, x0 = None, x1 = None, x2 = None, x3 = None):
		if x0 != None: self.val[0] = x0
		else: self.val[0] = 0.0
		if x1 != None: self.val[1] = x1
		else: self.val[1] = 0.0
		if x2 != None: self.val[2] = x2
		else: self.val[2] = 0.0
		if x3 != None: self.val[3] = x3
		else: self.val[3] = 0.0

	# mpfr_set_qd
	def mpfr_set_qd(self, prec = None):
		old_prec = gmpy2.get_context().precision
		if prec != None:
			gmpy2.get_context().precision = prec
		r = gmpy2.mpfr('0.0')
#		with gmpy2.ieee(64) as ctx:
		mpfr_val = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val[0] = gmpy2.mpfr(self.val[0])
		mpfr_val[1] = gmpy2.mpfr(self.val[1])
		mpfr_val[2] = gmpy2.mpfr(self.val[2])
		mpfr_val[3] = gmpy2.mpfr(self.val[3])

		r = mpfr_val[0] + mpfr_val[1] + mpfr_val[2] + mpfr_val[3]

		gmpy2.get_context().precision = old_prec

		return r

	# print
	def __str__(self):
		tmp = self.mpfr_set_qd(53 * 4)
		return str(tmp)

	# addition
	def __add__(self, y):
		ret = qd_float()
		librdd.rqd_add(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# iaddition
	def __iadd__(self, y):
		ret = qd_float()
		librdd.rqd_add(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# subtraction
	def __sub__(self, y):
		ret = qd_float()
		librdd.rqd_sub(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# multiplication
	def __mul__(self, y):
		ret = qd_float()
		librdd.rqd_mul(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret

	# division
	def __truediv__(self, y):
		ret = qd_float()
		librdd.rqd_div(
			ct.byref(ret.val),
			ct.byref(self.val),
			ct.byref(y.val)
		)
		return ret


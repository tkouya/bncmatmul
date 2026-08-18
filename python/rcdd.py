# test_rcdd.py: LIBRCDD test

# ref
# 1. https://qiita.com/kuboshu83/items/e76d5fdeac6132734a07
# 2. https://docs.python.org/ja/3/library/ctypes.html

import ctypes as ct
import gmpy2
import rdd # ddfloat, tdfloat, qdfloat

# MPFR library
libmpfr = ct.cdll.LoadLibrary('libmpfr.so')

# RDD library
librdd = ct.cdll.LoadLibrary('librdd.so')

# MPC library
libmpc = ct.cdll.LoadLibrary('libmpc.so')

# RCDD library
librcdd = ct.cdll.LoadLibrary('librcdd.so');

# mpc_get_cdd
def mpc_get_cdd(mpc_val):
	cdd_ret = cdd_float()
	cdd_ret.val_re[0] = float(mpc_val.real)
	cdd_ret.val_re[1] = float(mpc_val.real - gmpy2.mpfr(cdd_ret.val_re[0]))
	return cdd_ret

# mpc_get_ctd
def mpc_get_ctd(mpc_val):
	ctd_ret = ctd_float()
	ctd_ret.val_re[0] = float(mpc_val.real)
	ctd_ret.val_re[1] = float(mpc_val.real - gmpy2.mpfr(ctd_ret.val_re[0]))
	ctd_ret.val_re[2] = float(mpc_val.real - gmpy2.mpfr(ctd_ret.val_re[0]) - gmpy2.mpfr(ctd_ret.val_re[1]))
	ctd_ret.val_im[0] = float(mpc_val.imag)
	ctd_ret.val_im[1] = float(mpc_val.imag - gmpy2.mpfr(ctd_ret.val_im[0]))
	ctd_ret.val_im[2] = float(mpc_val.imag - gmpy2.mpfr(ctd_ret.val_im[0]) - gmpy2.mpfr(ctd_ret.val_im[1]))
	return ctd_ret

# mpc_get_cqd
def mpc_get_cqd(mpc_val):
	cqd_ret = cqd_float()
	cqd_ret.val_re[0] = float(mpc_val.real)
	cqd_ret.val_re[1] = float(mpc_val.real - gmpy2.mpfr(cqd_ret.val_re[0]))
	cqd_ret.val_re[2] = float(mpc_val.real - gmpy2.mpfr(cqd_ret.val_re[0]) - gmpy2.mpfr(cqd_ret.val_re[1]))
	cqd_ret.val_re[3] = float(mpc_val.real - gmpy2.mpfr(cqd_ret.val_re[0]) - gmpy2.mpfr(cqd_ret.val_re[1]) - gmpy2.mpfr(cqd_ret.val_re[2]))
	return cqd_ret

# cdd_float
class cdd_float(ct.Structure):
	_fields_ = [
		('val_re', ct.c_double * 2),
	    ('val_im', ct.c_double * 2)
    ]

	# constructor
	def __init__(self, re_0 = None, re_1 = None, im_0 = None, im_1 = None):
		librcdd._bnc_rcdd_set0(ct.byref(self))
		if re_0 != None: self.val_re[0] = re_0
		else: self.val_re[0] = 0.0
		if re_1 != None: self.val_re[1] = re_1
		else: self.val_re[1] = 0.0
		if im_0 != None: self.val_im[0] = im_0
		else: self.val_im[0] = 0.0
		if im_1 != None: self.val_im[1] = im_1
		else: self.val_im[1] = 0.0

	# mpc_set_cdd
	def mpc_set_cdd(self, prec = None):
		old_prec = gmpy2.get_context().precision
		if prec != None:
			gmpy2.get_context().precision = prec

		#r = gmpy2.mpc(0, 0)

		mpfr_val_re = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val_re[0] = gmpy2.mpfr(self.val_re[0])
		mpfr_val_re[1] = gmpy2.mpfr(self.val_re[1])

		mpfr_val_im = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val_im[0] = gmpy2.mpfr(self.val_im[0])
		mpfr_val_im[1] = gmpy2.mpfr(self.val_im[1])
		#r.real = mpfr_val_re[0] + mpfr_val_re[1]
		#r.imag = mpfr_val_im[0] + mpfr_val_im[1]
		r = gmpy2.mpc(
			mpfr_val_re[0] + mpfr_val_re[1],
			mpfr_val_im[0] + mpfr_val_im[1]
		)

		gmpy2.get_context().precision = old_prec

		return r

	# print
	def __str__(self):
		tmp = self.mpc_set_cdd(53 * 2)
		return str(tmp)

	# addition
	def __add__(self, y):
		ret = cdd_float()
		librcdd._bnc_rcdd_add(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# subtraction
	def __sub__(self, y):
		ret = cdd_float()
		librcdd._bnc_rcdd_sub(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# multiplication
	def __mul__(self, y):
		ret = cdd_float()
		librcdd._bnc_rcdd_mul(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# division
	def __truediv__(self, y):
		ret = cdd_float()
		librcdd._bnc_rcdd_div(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

# ctd_float
class ctd_float(ct.Structure):
	_fields_ = [
		('val_re', ct.c_double * 3),
		('val_im', ct.c_double * 3)
	]

	# constructor
	def __init__(self, re_0 = None, re_1 = None, re_2 = None, im_0 = None, im_1 = None, im_2 = None):
		librcdd._bnc_rctd_set0(ct.byref(self))
		if re_0 != None: self.val_re[0] = re_0
		if re_1 != None: self.val_re[1] = re_1
		if re_2 != None: self.val_re[2] = re_2
		if im_0 != None: self.val_im[0] = im_0
		if im_1 != None: self.val_im[1] = im_1
		if im_1 != None: self.val_im[2] = im_2

	# mpfr_set_td
	def mpc_set_ctd(self, prec = None):
		old_prec = gmpy2.get_context().precision
		if prec != None:
			gmpy2.get_context().precision = prec

		r = gmpy2.mpfc('0.0', '0.0')

		mpfr_val_re = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val_re[0] = gmpy2.mpfr(self.val_re[0])
		mpfr_val_re[1] = gmpy2.mpfr(self.val_re[1])
		mpfr_val_re[2] = gmpy2.mpfr(self.val_re[2])

		mpfr_val_im = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val_im[0] = gmpy2.mpfr(self.val_im[0])
		mpfr_val_im[1] = gmpy2.mpfr(self.val_im[1])
		mpfr_val_im[2] = gmpy2.mpfr(self.val_im[2])

		r.real = mpfr_val_re[0] + mpfr_val_re[1] + mpfr_val_re[2]
		r.imag = mpfr_val_im[0] + mpfr_val_im[1] + mpfr_val_im[2]

		gmpy2.get_context().precision = old_prec

		return r

	# print
	def __str__(self):
		tmp = self.mpc_set_ctd(53 * 3)
		return str(tmp)

	# addition
	def __add__(self, y):
		ret = ctd_float()
		librcdd._bnc_rctd_add(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# subtraction
	def __sub__(self, y):
		ret = ctd_float()
		librcdd._bnc_rctd_sub(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# multiplication
	def __mul__(self, y):
		ret = ctd_float()
		librcdd._bnc_rctd_mul(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# division
	def __truediv__(self, y):
		ret = ctd_float()
		librcdd._bnc_rctd_div(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

# cqd_float
class cqd_float(ct.Structure):
	_fields_ = [
		('val_re', ct.c_double * 4),
		('val_im', ct.c_double * 4)
	]

	# constructor
	def __init__(self, re_0 = None, re_1 = None, re_2 = None, re_3 = None, im_0 = None, im_1 = None, im_2 = None, im_3 = None):
		librcdd._bnc_rcqd_set0(ct.byref(self))
		if re_0 != None: self.val_re[0] = re_0
		if re_1 != None: self.val_re[1] = re_1
		if re_2 != None: self.val_re[2] = re_2
		if re_3 != None: self.val_re[3] = re_3
		if im_0 != None: self.val_im[0] = im_0
		if im_1 != None: self.val_im[1] = im_1
		if im_2 != None: self.val_im[2] = im_2
		if im_3 != None: self.val_im[3] = im_3


	# mpfr_set_qd
	def mpc_set_cqd(self, prec = None):
		old_prec = gmpy2.get_context().precision
		if prec != None:
			gmpy2.get_context().precision = prec

		r = gmpy2.mpfc('0.0', '0.0')

		mpfr_val_re = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val_re[0] = gmpy2.mpfr(self.val_re[0])
		mpfr_val_re[1] = gmpy2.mpfr(self.val_re[1])
		mpfr_val_re[2] = gmpy2.mpfr(self.val_re[2])
		mpfr_val_re[3] = gmpy2.mpfr(self.val_re[3])

		mpfr_val_im = [gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0'), gmpy2.mpfr('0.0')]
		mpfr_val_im[0] = gmpy2.mpfr(self.val_im[0])
		mpfr_val_im[1] = gmpy2.mpfr(self.val_im[1])
		mpfr_val_im[2] = gmpy2.mpfr(self.val_im[2])
		mpfr_val_im[3] = gmpy2.mpfr(self.val_im[3])

		r.real = mpfr_val_re[0] + mpfr_val_re[1] + mpfr_val_re[2] + mpfr_val_re[3]
		r.imag = mpfr_val_im[0] + mpfr_val_im[1] + mpfr_val_im[2] + mpfr_val_im[3]

		gmpy2.get_context().precision = old_prec

		return r

	# print
	def __str__(self):
		tmp = self.mpc_set_cqd(53 * 4)
		return str(tmp)

	# addition
	def __add__(self, y):
		ret = cqd_float()
		librcdd._bnc_rcqd_add(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# subtraction
	def __sub__(self, y):
		ret = cqd_float()
		librcdd._bnc_rcqd_sub(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# multiplication
	def __mul__(self, y):
		ret = cqd_float()
		librcdd._bnc_rcqd_mul(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret

	# division
	def __truediv__(self, y):
		ret = cqd_float()
		librcdd._bnc_rcqd_div(
			ct.byref(ret),
			ct.byref(self),
			ct.byref(y)
		)
		return ret


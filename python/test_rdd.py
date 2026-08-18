# test_rdd.py: LIBRDD test

# ref
# 1. https://qiita.com/kuboshu83/items/e76d5fdeac6132734a07
# 2. https://docs.python.org/ja/3/library/ctypes.html

import ctypes as ct
import rdd
import math
import gmpy2

# gmpy2
print(gmpy2.get_context())
gmpy2.get_context().precision = 1024 # in bits
print(gmpy2.get_context())
#print(gmpy2.sqrt(gmpy2.mpfr(2)))

a = math.sqrt(float(2.0))
#mpfr_a = gmpy2.mpfr(a)
mpfr_a = gmpy2.sqrt(gmpy2.mpfr(2))
mpfr_b = gmpy2.sqrt(gmpy2.mpfr(3))
#print('a = ', a)
#print('mpfr_a = ', mpfr_a)
#print('mpfr_b = ', mpfr_b)

# RDD library
librdd = ct.cdll.LoadLibrary('librdd.so');

# start
librdd.rdd_start();

# dd_float
dd_a = rdd.dd_float(0.0, 0.0)
print('dd_a = ', dd_a)
dd_a = rdd.mpfr_get_dd(mpfr_a)
dd_b = rdd.mpfr_get_dd(mpfr_b)
print('dd_a = ', dd_a)
print('dd_b = ', dd_b)
print('dd_a + dd_b = ', dd_a + dd_b)
print('dd_a - dd_b = ', dd_a - dd_b)
print('dd_a * dd_b = ', dd_a * dd_b)
print('dd_a / dd_b = ', dd_a / dd_b)

# td_float
td_a = rdd.td_float(0.0, 0.0, 0.0)
print('td_a = ', td_a)
td_a = rdd.mpfr_get_td(mpfr_a)
td_b = rdd.mpfr_get_td(mpfr_b)
print('td_a = ', td_a)
print('td_b = ', td_b)
print('td_a + td_b = ', td_a + td_b)
print('td_a - td_b = ', td_a - td_b)
print('td_a * td_b = ', td_a * td_b)
print('td_a / td_b = ', td_a / td_b)

# qd_float
qd_a = rdd.qd_float(0.0, 0.0, 0.0)
print('qd_a = ', qd_a)
qd_a = rdd.mpfr_get_qd(mpfr_a)
print('qd_a = ', qd_a)
qd_b = rdd.mpfr_get_qd(mpfr_b)
print('qd_b = ', qd_b)
print('qd_a + qd_b = ', qd_a + qd_b)
print('qd_a - qd_b = ', qd_a - qd_b)
print('qd_a * qd_b = ', qd_a * qd_b)
print('qd_a / qd_b = ', qd_a / qd_b)


librdd.rdd_end();

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define USE_rcdd_FUNCTIONS
#define USE_rctd_FUNCTIONS
#define USE_rcqd_FUNCTIONS
//#include "rdd.h"
#include "rcdd.h"

// DD print(no appending CR)
extern void rcdd_out_str(cddfloat *val);

// QD print(no appending CR)
extern void rcqd_out_st(cqdfloat *val);

// DD
void _bnc_set0_cdd(cddfloat *val) { rcdd_set0(val); }
void _bnc_rcdd_set0(cddfloat *val) { rcdd_set0(val); }
void _bnc_rcdd_add(cddfloat *ret, cddfloat *a, cddfloat *b) {  rcdd_add(ret, a, b); }
void _bnc_rcdd_sub(cddfloat *ret, cddfloat *a, cddfloat *b) {  rcdd_sub(ret, a, b); }
void _bnc_rcdd_mul(cddfloat *ret, cddfloat *a, cddfloat *b) {  rcdd_mul(ret, a, b); }
void _bnc_rcdd_div(cddfloat *ret, cddfloat *a, cddfloat *b) {  rcdd_div(ret, a, b); }
void _bnc_rcdd_set(cddfloat *ret, cddfloat *org) {  rcdd_set(ret, org); }
void _bnc_rcdd_neg(cddfloat *ret, cddfloat *a) {  rcdd_neg(ret, a); }
void _bnc_rcdd_abs(ddfloat *ret, cddfloat *a) {  rcdd_abs(ret, a); }

// TD
void _bnc_set0_ctd(ctdfloat *val) {  rctd_set0(val); }
void _bnc_rctd_set0(ctdfloat *val) {  rctd_set0(val); }
void _bnc_rctd_add(ctdfloat *ret, ctdfloat *a, ctdfloat *b) {  rctd_add(ret, a, b); }
void _bnc_rctd_sub(ctdfloat *ret, ctdfloat *a, ctdfloat *b) {  rctd_sub(ret, a, b); }
void _bnc_rctd_mul(ctdfloat *ret, ctdfloat *a, ctdfloat *b) {  rctd_mul(ret, a, b); }
void _bnc_rctd_div(ctdfloat *ret, ctdfloat *a, ctdfloat *b) {  rctd_div(ret, a, b); }
void _bnc_rctd_set(ctdfloat *ret, ctdfloat *org) {  rctd_set(ret, org); }
void _bnc_rctd_neg(ctdfloat *ret, ctdfloat *a) {  rctd_neg(ret, a); }
void _bnc_rctd_abs(tdfloat *ret, ctdfloat *a) {  rctd_abs(ret, a); }

// QD 
void _bnc_set0_cqd(cqdfloat *val) {  rcqd_set0(val); }
void _bnc_rcqd_set0(cqdfloat *val) {  rcqd_set0(val); }
void _bnc_rcqd_add(cqdfloat *ret, cqdfloat *a, cqdfloat *b) {  rcqd_add(ret, a, b); }
void _bnc_rcqd_sub(cqdfloat *ret, cqdfloat *a, cqdfloat *b) {  rcqd_sub(ret, a, b); }
void _bnc_rcqd_mul(cqdfloat *ret, cqdfloat *a, cqdfloat *b) {  rcqd_mul(ret, a, b); }
void _bnc_rcqd_div(cqdfloat *ret, cqdfloat *a, cqdfloat *b) {  rcqd_div(ret, a, b); }
void _bnc_rcqd_set(cqdfloat *ret, cqdfloat *org) {  rcqd_set(ret, org); }
void _bnc_rcqd_neg(cqdfloat *ret, cqdfloat *a) {  rcqd_neg(ret, a); }
void _bnc_rcqd_abs(qdfloat *ret, cqdfloat *a) {  rcqd_abs(ret, a); }

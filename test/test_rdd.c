// gcc test_rdd.c -lmpfr -lgmp -lm
#include <stdio.h>
#include <math.h>

#include "mpfr.h"
#include "mpfr_dtq_sd.h"

#include "rdd.h"

int main()
{
	double dd_a[DDSIZE], dd_b[DDSIZE], dd_c[DDSIZE], dd_d[DDSIZE], dd_e[DDSIZE], dd_f[DDSIZE];
	double td_a[TDSIZE], td_b[TDSIZE], td_c[TDSIZE], td_d[TDSIZE], td_e[TDSIZE], td_f[TDSIZE];
	double qd_a[QDSIZE], qd_b[QDSIZE], qd_c[QDSIZE], qd_d[QDSIZE], qd_e[QDSIZE], qd_f[QDSIZE];
	mpfr_t mp_a, mp_b, mp_f;

	mpfr_set_default_prec(53 * 5);
	mpfr_init(mp_a); mpfr_sqrt_ui(mp_a, 3UL, MPFR_RNDN);
	mpfr_init(mp_b); mpfr_sqrt_ui(mp_b, 2UL, MPFR_RNDN);
	mpfr_init(mp_f);
//	mpfr_printf("%RNe, %RNe\n", mp_a, mp_b);
	mpfr_div(mp_f, mp_a, mp_b, MPFR_RNDN);

	fpu_fix_start(NULL);

	// double precision
	//printf("dp_a = %25.17e\n",sqrt(3.0));
	printf("1 + 2 * DBL_EPSILON = %25.17e\n", 1 + 2 * DBL_EPSILON);
	printf("1 - 2 * DBL_EPSILON = %25.17e\n", 1 - 2 * DBL_EPSILON);

	// DD precision
	set0_dd(dd_a);
	rdd_set_ui(dd_a, 3UL);
//	rdd_sqrt(dd_a, dd_a);
	mpfr_get_dd(dd_a, mp_a, MPFR_RNDN);
	rdd_set_ui(dd_b, 2UL);
//	rdd_sqrt(dd_b, dd_b);
	mpfr_get_dd(dd_b, mp_b, MPFR_RNDN);
	rdd_mul(dd_c, dd_a, dd_b); // sloppy
	rdd_add(dd_d, dd_a, dd_b);
	rdd_sub(dd_e, dd_a, dd_b);
	rdd_div(dd_f, dd_a, dd_b);

	// Triple precision
	set0_td(td_a);
	rtd_set_ui(td_a, 3UL);
//	rtd_sqrt(td_a, td_a);
	mpfr_get_td(td_a, mp_a, MPFR_RNDN);
	rtd_set_ui(td_b, 2UL);
//	rtd_sqrt(td_b, td_b);
	mpfr_get_td(td_b, mp_b, MPFR_RNDN);
	rtd_mul(td_c, td_a, td_b); // sloppy
	//c_td_mul_accurate(td_a, td_b, td_c);
	rtd_add(td_d, td_a, td_b);
	rtd_sub(td_e, td_a, td_b);
	rtd_div(td_f, td_a, td_b);

	// Quadruple precision
	set0_qd(qd_a);
	rqd_set_ui(qd_a, 3UL);
	//rqd_sqrt(qd_a, qd_a);
	mpfr_get_qd(qd_a, mp_a, MPFR_RNDN);
	rqd_set_ui(qd_b, 2UL);
	//rqd_sqrt(qd_b, qd_b);
	mpfr_get_qd(qd_b, mp_b, MPFR_RNDN);
	rqd_mul(qd_c, qd_a, qd_b);
	rqd_add(qd_d, qd_a, qd_b);
	rqd_sub(qd_e, qd_a, qd_b);
	rqd_div(qd_f, qd_a, qd_b);

	//printf("dd_a = "); rdd_out_str(dd_a); printf("\n");
	printf("d   (sqrt)=%24.17e\n", sqrt(3.0));
	printf("dd_a(sqrt)= "); rdd_out_str(dd_a); printf("\n");
	printf("td_a(sqrt)= "); rtd_out_str(td_a); printf("\n");
	printf("qd_a(sqrt)= "); rqd_out_str(qd_a); printf("\n");
	mpfr_printf("mp_a(sqrt)= %RNe\n", mp_a);
	printf("d   (sqrt)=%24.17e\n", sqrt(2.0));
	printf("dd_b(sqrt)= "); rdd_out_str(dd_b); printf("\n");
	printf("td_b(sqrt)= "); rtd_out_str(td_b); printf("\n");
	printf("qd_b(sqrt)= "); rqd_out_str(qd_b); printf("\n");
	mpfr_printf("mp_b(sqrt)= %RNe\n", mp_b);
	printf("dd_c(mul) = "); rdd_out_str(dd_c); printf("\n");
	printf("td_c(mul) = "); rtd_out_str(td_c); printf("\n");
	printf("qd_c(mul) = "); rqd_out_str(qd_c); printf("\n");
	printf("dd_d(add) = "); rdd_out_str(dd_d); printf("\n");
	printf("td_d(add) = "); rtd_out_str(td_d); printf("\n");
	printf("qd_d(add) = "); rqd_out_str(qd_d); printf("\n");
	printf("dd_e(sub) = "); rdd_out_str(dd_e); printf("\n");
	printf("td_e(sub) = "); rtd_out_str(td_e); printf("\n");
	printf("qd_e(sub) = "); rqd_out_str(qd_e); printf("\n");
	printf("dd_f(div) = "); rdd_out_str(dd_f); printf("\n");
	printf("td_f(div) = "); rtd_out_str(td_f); printf("\n");
	printf("qd_f(div) = "); rqd_out_str(qd_f); printf("\n");
	mpfr_printf("mp_f(div) = %Re\n", mp_f);

	mpfr_clear(mp_a);
	mpfr_clear(mp_b);
	mpfr_clear(mp_f);

	return 0;
}

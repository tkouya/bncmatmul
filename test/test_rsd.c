// test_rds.c: double-sigle, triple-single, quad-single prec. arithmetic
// 2021-04-23(Fri): Tomonori Kouya

// How to compile: $ gcc -DUSE_MPFR -DUSE_GMP test_rsd.c mpfr_dtq_sd.c -lmpfr -lgmp -lm
#include <stdio.h>
#include <math.h>

#include "rds.h" // rds_*, rts_*, rqs_* functions
#include "mpfr_dtq_sd.h" // mpfr_[gs]et_[dtq][sd] and r[dtq][sd]_out_str functions

int main()
{
	float ds_a[DSSIZE], ds_b[DSSIZE], ds_c[DSSIZE], ds_d[DSSIZE], ds_e[DSSIZE], ds_f[DSSIZE];
	float ts_a[TSSIZE], ts_b[TSSIZE], ts_c[TSSIZE], ts_d[TSSIZE], ts_e[TSSIZE], ts_f[TSSIZE];
	float qs_a[QSSIZE], qs_b[QSSIZE], qs_c[QSSIZE], qs_d[QSSIZE], qs_e[QSSIZE], qs_f[QSSIZE];
    double d_a, d_b, d_c, d_d, d_e, d_f;
	mpfr_t mp_a, mp_b, mp_c, mp_d, mp_e, mp_f;

	mpfr_set_default_prec(24 * 5);
	mpfr_init(mp_a); mpfr_sqrt_ui(mp_a, 3UL, MPFR_RNDN);
	mpfr_init(mp_b); mpfr_sqrt_ui(mp_b, 2UL, MPFR_RNDN);
	mpfr_init(mp_c); mpfr_init(mp_d); mpfr_init(mp_e); mpfr_init(mp_f);
//	mpfr_printf("%RNe, %RNe\n", mp_a, mp_b);

    // MPFR precision
	mpfr_add(mp_c, mp_a, mp_b, MPFR_RNDN);
    mpfr_sub(mp_d, mp_a, mp_b, MPFR_RNDN);
    mpfr_mul(mp_e, mp_a, mp_b, MPFR_RNDN);
	mpfr_div(mp_f, mp_a, mp_b, MPFR_RNDN);

	//fpu_fix_start(NULL);

    // double precision
    d_a = sqrt(3.0);
    d_b = sqrt(2.0);
    d_c = d_a + d_b;
    d_d = d_a - d_b;
    d_e = d_a * d_b;
    d_f = d_a / d_b;

	// single precision
	//printf("sp_a = %15.7e\n", sqrtf((float)3.0));
	printf("1 + 2 * FLT_EPSILON = %15.7e\n", 1 + 2 * FLT_EPSILON);
	printf("1 - 2 * FLT_EPSILON = %15.7e\n", 1 - 2 * FLT_EPSILON);

	// DS precision
	set0_ds(ds_a);
	rds_set_ui(ds_a, 3UL);
//	rds_sqrt(ds_a, dd_a);
	mpfr_get_ds(ds_a, mp_a, MPFR_RNDN);
	rds_set_ui(ds_b, 2UL);
//	rdd_sqrt(ds_b, ds_b);
	mpfr_get_ds(ds_b, mp_b, MPFR_RNDN);
	rds_add(ds_c, ds_a, ds_b);
	rds_sub(ds_d, ds_a, ds_b);
	rds_mul(ds_e, ds_a, ds_b);
	rds_div(ds_f, ds_a, ds_b);

	// TS precision
	set0_ts(ts_a);
	rts_set_ui(ts_a, 3UL);
//	rts_sqrt(ds_a, dd_a);
	mpfr_get_ts(ts_a, mp_a, MPFR_RNDN);
	rts_set_ui(ts_b, 2UL);
//	rts_sqrt(ts_b, ts_b);
	mpfr_get_ts(ts_b, mp_b, MPFR_RNDN);
	rts_add(ts_c, ts_a, ts_b);
	rts_sub(ts_d, ts_a, ts_b);
	rts_mul(ts_e, ts_a, ts_b);
	rts_div(ts_f, ts_a, ts_b);

	// QS precision
	set0_qs(qs_a);
	rqs_set_ui(qs_a, 3UL);
//	rqs_sqrt(qs_a, qs_a);
	mpfr_get_qs(qs_a, mp_a, MPFR_RNDN);
	rqs_set_ui(qs_b, 2UL);
//	rqs_sqrt(qs_b, qs_b);
	mpfr_get_qs(qs_b, mp_b, MPFR_RNDN);
	rqs_add(qs_c, qs_a, qs_b);
	rqs_sub(qs_d, qs_a, qs_b);
	rqs_mul(qs_e, qs_a, qs_b);
	rqs_div(qs_f, qs_a, qs_b);


	printf(" d_a(sqrt)=%24.17e\n", d_a);
	printf("ds_a(sqrt)= "); rds_out_str(ds_a); printf("\n");
	printf("ts_a(sqrt)= "); rts_out_str(ts_a); printf("\n");
	printf("qs_a(sqrt)= "); rqs_out_str(qs_a); printf("\n");
    mpfr_printf("mp_a(sqrt)= %RNe\n", mp_a);

	printf(" d_b(sqrt)=%24.17e\n", d_b);
	printf("ds_b(sqrt)= "); rds_out_str(ds_b); printf("\n");
	printf("ts_b(sqrt)= "); rts_out_str(ts_b); printf("\n");		
    printf("qs_b(sqrt)= "); rqs_out_str(qs_b); printf("\n");
    mpfr_printf("mp_b(sqrt)= %RNe\n", mp_b);

	printf(" d_c(add) =%24.17e\n", d_c);
	printf("ds_c(add) = "); rds_out_str(ds_c); printf("\n");
	printf("ts_c(add) = "); rts_out_str(ts_c); printf("\n");	
	printf("qs_c(add) = "); rqs_out_str(qs_c); printf("\n");	
    mpfr_printf("mp_c(add) = %RNe\n", mp_c);

	printf(" d_d(sub) =%24.17e\n", d_d);
	printf("ds_d(sub) = "); rds_out_str(ds_d); printf("\n");
	printf("ts_d(sub) = "); rts_out_str(ts_d); printf("\n");
	printf("qs_c(sub) = "); rqs_out_str(qs_d); printf("\n");
    mpfr_printf("mp_d(sub) = %RNe\n", mp_d);

	printf(" d_e(mul) =%24.17e\n", d_e);
	printf("ds_e(mul) = "); rds_out_str(ds_e); printf("\n");
	printf("ts_e(mul) = "); rts_out_str(ts_e); printf("\n");		
    printf("qs_e(mul) = "); rqs_out_str(qs_e); printf("\n");
    mpfr_printf("mp_e(mul) = %RNe\n", mp_e);

	printf(" d_f(div) =%24.17e\n", d_f);
	printf("ds_f(div) = "); rds_out_str(ds_f); printf("\n");
	printf("ts_f(div) = "); rts_out_str(ts_f); printf("\n");		
	printf("qs_f(div) = "); rqs_out_str(qs_f); printf("\n");    
    mpfr_printf("mp_f(div) = %Re\n", mp_f);

	// set_d
	rds_set_d(ds_a, d_a);
	rts_set_d(ts_a, d_a);
	rqs_set_d(qs_a, d_a);
	mpfr_set_d(mp_a, d_a, MPFR_RNDN);
	printf(" d_a =%24.17e\n", d_a);
	printf("ds_a = "); rds_out_str(ds_a); printf("\n");
	printf("ts_a = "); rts_out_str(ts_a); printf("\n");
	printf("qs_a = "); rqs_out_str(qs_a); printf("\n");
    mpfr_printf("mp_a = %Re\n", mp_a);

	mpfr_clear(mp_a);
	mpfr_clear(mp_b);
	mpfr_clear(mp_c);
	mpfr_clear(mp_d);
	mpfr_clear(mp_e);
	mpfr_clear(mp_f);

	return 0;
}


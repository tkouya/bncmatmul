/********************************************************************************/
/* test_spmm.c:                                                                 */
/* Copyright (C) 2024 Tomonori Kouya                                            */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bncmatmul.h"
#include "get_secv.h" // get_secv
//#include "bnc.h"
//#include "bncsparse.h"
//#include "bncmm.h"

/* Downloaded file from UF Sparse Matrix Collection */
#define MATRIX_MM_FILE "cage3/cage3.mtx"

#define MAXTIMES 1000
#define PREC 512

// get_dvector
void get_dvector(DVector dvec_x)
{
	long int i;

	for(i = 0; i < dvec_x->dim; i++)
	    set_dvector_i(dvec_x, i, (double)(i + 1) * sqrt(2.0));
}

// get_ddvector
void get_ddvector(DDVector vec_x)
{
	long int i;
	double tmp[DDSIZE];

	for(i = 0; i < vec_x->dim; i++)
	{
		rdd_sqrt_ui(tmp, 2UL); // OK!
		rdd_mul_ui(tmp, tmp, (unsigned long)(i + 1));
	    set_ddvector_i(vec_x, i, tmp); //(double)(i + 1) * sqrt(2.0));
	}
}

// get_tdvector
void get_tdvector(TDVector vec_x)
{
	long int i;
	double tmp[TDSIZE];
	mpf_t sqrt2;

	mpf_init2(sqrt2, 159);
	mpf_sqrt_ui(sqrt2, 2UL);

	for(i = 0; i < vec_x->dim; i++)
	{
		//rtd_sqrt_ui(tmp, 2UL);
		mpf_get_td(tmp, sqrt2);
		//rtd_out_str(tmp);
		rtd_mul_ui(tmp, tmp, (unsigned long)(i + 1));
		//rtd_out_str(tmp);
	    set_tdvector_i(vec_x, i, tmp); //(double)(i + 1) * sqrt(2.0));
		//rtd_out_str(tmp);
	}

	mpf_clear(sqrt2);
}

// get_qdvector
void get_qdvector(QDVector vec_x)
{
	long int i;
	double tmp[QDSIZE], tmp2[QDSIZE];
	mpf_t sqrt2;

	mpf_init2(sqrt2, 212);
	mpf_sqrt_ui(sqrt2, 2UL);

	for(i = 0; i < vec_x->dim; i++)
	{
		//rqd_sqrt_ui(tmp, 2UL);
		mpf_get_qd(tmp, sqrt2);
		//rqd_set_ui(tmp2, (unsigned long)(i + 1));
		rqd_mul_ui(tmp, tmp, (unsigned long)(i + 1));
	    set_qdvector_i(vec_x, i, tmp); //(double)(i + 1) * sqrt(2.0));
	}

	mpf_clear(sqrt2);
}

#ifdef USE_GMP
// get_mpfvector
void get_mpfvector(MPFVector vec_x)
{
	long int i;
	mpf_t vi;

	mpf_init2(vi, vec_x->prec);

	for(i = 0; i < vec_x->dim; i++)
	{
		mpf_sqrt_ui(vi, 2UL); mpf_mul_ui(vi, vi, (unsigned long)(i + 1));
	    set_mpfvector_i(vec_x, i, vi); // (double)(i + 1) * sqrt(2.0));
		//mpfr_printf("%5d %25.17RNe\n", i, vi);
	}

	mpf_clear(vi);
}
#endif // USE_GMP

int main(void)
{
	long int dim, total_index;
	long int maxtimes = MAXTIMES;
	DMatrix da;
	DRSMatrix da_sp;
    DVector dvec_x, dvec_b, dvec_xt, dvec_tb;
    DDMatrix dda;
    DDRSMatrix dda_sp;
    DDVector ddvec_x, ddvec_b;
    TDMatrix tda;
    TDRSMatrix tda_sp;
    TDVector tdvec_x, tdvec_b;
    QDMatrix qda;
    QDRSMatrix qda_sp;
    QDVector qdvec_x, qdvec_b;
    double dtmp, ddtmp[DDSIZE], tdtmp[TDSIZE], qdtmp[QDSIZE];
	double dnorm_av[2], ddnorm_av[2][DDSIZE], tdnorm_av[2][TDSIZE], qdnorm_av[2][QDSIZE];
	double dnorm_atv[2], ddnorm_atv[2][DDSIZE], tdnorm_atv[2][TDSIZE], qdnorm_atv[2][QDSIZE];
	double start, dtime[5], dtime_sp[5];
	long int itimes_d[5], itimes_d_sp[5];
	long int i, j;

#ifdef USE_GMP
	MPFMatrix mpfa;
	MPFMatrix mpfa2;
	MPFMatrix mpfa3;
	MPFMatrix mpfa4;
	MPFMatrix mpfa5;

	MPFRSMatrix mpfa_sp;
	MPFRSMatrix mpfa2_sp;
	MPFRSMatrix mpfa3_sp;
	MPFRSMatrix mpfa4_sp;
	MPFRSMatrix mpfa5_sp;
	MPFVector mpfvec_x, mpfvec_b;
	mpf_t mpfnorm_av[2], mpfnorm_atv[2];
#endif // USE_GMP

//goto MPFR;

/* Double */
	/* initialize & get problem */

	// Read MM file as SPARSE matrix
    printf("Reading %s...\n", MATRIX_MM_FILE);
	da_sp = init_drsmatrix_readMMcoordinate(MATRIX_MM_FILE);
    print_drsmatrix(da_sp);

	// A * v
	dvec_x = init_dvector(da_sp->col_dim);
	dvec_b = init_dvector(da_sp->row_dim);

	get_dvector(dvec_x);
	mul_drsmatrix_dvec(dvec_b, da_sp, dvec_x);
	dnorm_av[0] = norm2_dvector(dvec_b);
	print_dvector(dvec_b);

	get_dvector(dvec_b);
	mul_drsmatrixt_dvec(dvec_x, da_sp, dvec_b);
	dnorm_atv[0] = norm2_dvector(dvec_x);
	print_dvector(dvec_x);

	free_dvector(dvec_x);
	free_dvector(dvec_b);
	free_drsmatrix(da_sp);

	// Read MM file as DENSE matrix
    printf("Reading %s...\n", MATRIX_MM_FILE);
	da = init_dmatrix_readMMcoordinate(MATRIX_MM_FILE);
    //print_dmatrix(da);

	// A * v
	dvec_x = init_dvector(da->col_dim);
	dvec_b = init_dvector(da->row_dim);

	get_dvector(dvec_x);
	mul_dmatrix_dvec(dvec_b, da, dvec_x);
	dnorm_av[1] = norm2_dvector(dvec_b);
	print_dvector(dvec_b);

	get_dvector(dvec_b);
	mul_dmatrixt_dvec(dvec_x, da, dvec_b);
	dnorm_atv[1] = norm2_dvector(dvec_x);
	print_dvector(dvec_x);

	free_dvector(dvec_x);
	free_dvector(dvec_b);
	free_dmatrix(da);

	// print norm
	printf("||DA_sp   * V||_2 = %25.17e\n", dnorm_av[0]);
	printf("||DA      * V||_2 = %25.17e\n", dnorm_av[1]);
	printf("||DA_sp^T * V||_2 = %25.17e\n", dnorm_atv[0]);
	printf("||DA^T    * V||_2 = %25.17e\n", dnorm_atv[1]);

MPFR:
#ifdef USE_GMP
/* MPF */
	set_bnc_default_prec(PREC);

	mpf_init(mpfnorm_av[0]);
	mpf_init(mpfnorm_av[1]);
	mpf_init(mpfnorm_atv[0]);
	mpf_init(mpfnorm_atv[1]);

	/* load */
	// Read MM file as SPARSE matrix
    printf("Reading %s...\n", MATRIX_MM_FILE);
	mpfa_sp = init_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE);
    //print_mpfrsmatrix(mpfa_sp);

	// A * v
	mpfvec_x = init_mpfvector(mpfa_sp->col_dim);
	mpfvec_b = init_mpfvector(mpfa_sp->row_dim);

	get_mpfvector(mpfvec_x);
	mul_mpfrsmatrix_mpfvec(mpfvec_b, mpfa_sp, mpfvec_x);
	norm2_mpfvector(mpfnorm_av[0], mpfvec_b);

	get_mpfvector(mpfvec_b);
	mul_mpfrsmatrixt_mpfvec(mpfvec_x, mpfa_sp, mpfvec_b);
	norm2_mpfvector(mpfnorm_atv[0], mpfvec_x);

	free_mpfvector(mpfvec_x);
	free_mpfvector(mpfvec_b);
	free_mpfrsmatrix(mpfa_sp);

	// Read MM file as DENSE matrix
    printf("Reading %s...\n", MATRIX_MM_FILE);
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
    //print_mpfmatrix(mpfa);
    //free_mpfmatrix(mpfa);
	// A * v
	mpfvec_x = init_mpfvector(mpfa->col_dim);
	mpfvec_b = init_mpfvector(mpfa->row_dim);

	get_mpfvector(mpfvec_x);
	mul_mpfmatrix_mpfvec(mpfvec_b, mpfa, mpfvec_x);
	norm2_mpfvector(mpfnorm_av[1], mpfvec_b);

	get_mpfvector(mpfvec_b);
	mul_mpfmatrixt_mpfvec(mpfvec_x, mpfa, mpfvec_b);
	norm2_mpfvector(mpfnorm_atv[1], mpfvec_x);

	free_mpfvector(mpfvec_x);
	free_mpfvector(mpfvec_b);
	free_mpfmatrix(mpfa);

	// print norm
	mpfr_printf("||MPFA_sp   * V||_2 = %75.65RNe\n", mpfnorm_av[0]);
	mpfr_printf("||MPFA      * V||_2 = %75.65RNe\n", mpfnorm_av[1]);
	mpfr_printf("||MPFA_sp^T * V||_2 = %75.65RNe\n", mpfnorm_atv[0]);
	mpfr_printf("||MPFA^T    * V||_2 = %75.65RNe\n", mpfnorm_atv[1]);

	mpf_clear(mpfnorm_av[0]);
	mpf_clear(mpfnorm_av[1]);
	mpf_clear(mpfnorm_atv[0]);
	mpf_clear(mpfnorm_atv[1]);

	//goto QD;

// -------------
// Double-double
// -------------
DD:
    // MPFRSMatrix -> DDRSMatrix
	mpfa_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
    //mpfa_sp = init_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE);
    printf("row_dim, nzero_total_num = %ld, %ld\n", mpfa_sp->row_dim, mpfa_sp->nzero_total_num);
/*   dda_sp = init_ddrsmatrix(mpfa_sp->row_dim, mpfa_sp->nzero_col_dim, mpfa_sp->nzero_total_num);
    printf("init_ddrsmatrix!\n");

    // MPFR -> DD
    total_index = 0;
	for(i = 0; i < dda_sp->row_dim; i++)
	{
		for(j = 0; j < dda_sp->nzero_col_dim[i]; j++)
		{
            mpf_get_dd(ddtmp, mpfa_sp->element[total_index]);
            dda_sp->element[0][total_index] = ddtmp[0];
            dda_sp->element[1][total_index] = ddtmp[1];
            dda_sp->nzero_index[i][j] = mpfa_sp->nzero_index[i][j];
			total_index++;
		}
	}
*/
    dda_sp = init_set_ddrsmatrix_mpfrsmatrix(mpfa_sp);
    print_ddrsmatrix(dda_sp);
	// A * v
	ddvec_x = init_ddvector(dda_sp->col_dim);
	ddvec_b = init_ddvector(dda_sp->row_dim);

	get_ddvector(ddvec_x);
	printf("start mul_ddrsmatrix_ddvec...");
	mul_ddrsmatrix_ddvec(ddvec_b, dda_sp, ddvec_x);
	printf("end\n");
	norm2_ddvector(ddnorm_av[0], ddvec_b);
	print_ddvector(ddvec_b);

	get_ddvector(ddvec_b);
	printf("start mul_ddrsmatrixt_ddvec...");
	mul_ddrsmatrixt_ddvec(ddvec_x, dda_sp, ddvec_b);
	printf("end\n");
	norm2_ddvector(ddnorm_atv[0], ddvec_x);
	print_ddvector(ddvec_x);

	free_ddvector(ddvec_x);
	free_ddvector(ddvec_b);
    free_ddrsmatrix(dda_sp);
    free_mpfrsmatrix(mpfa_sp);

	// Read MM file as DENSE matrix
    printf("Reading %s...\n", MATRIX_MM_FILE);
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	//mpfa = init_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE);
	dda = init_ddmatrix(mpfa->row_dim, mpfa->col_dim);
	subst_ddmatrix_mpfmat(dda, mpfa);
	//printf("Dense DDA = \n");
    //print_ddmatrix(dda);

	// A * v
	ddvec_x = init_ddvector(dda->col_dim);
	ddvec_b = init_ddvector(dda->row_dim);

	get_ddvector(ddvec_x);
	mul_ddmatrix_ddvec(ddvec_b, dda, ddvec_x);
	norm2_ddvector(ddnorm_av[1], ddvec_b);
	print_ddvector(ddvec_b);

	get_ddvector(ddvec_b);
	mul_ddmatrixt_ddvec(ddvec_x, dda, ddvec_b);
	norm2_ddvector(ddnorm_atv[1], ddvec_x);
	print_ddvector(ddvec_x);

	//print_ddvector(ddvec_x);
	//print_ddvector(ddvec_b);
	//printf("||DDA      * V||_2 = "); rdd_out_str(ddnorm_av[1]);  printf("\n");
	//printf("||DDA^T    * V||_2 = "); rdd_out_str(ddnorm_atv[1]); printf("\n");

	free_ddvector(ddvec_x);
	free_ddvector(ddvec_b);
	free_ddmatrix(dda);
	free_mpfmatrix(mpfa);

	// print norm
	printf("||DDA_sp   * V||_2 = "); rdd_out_str(ddnorm_av[0]);  printf("\n");
	printf("||DDA      * V||_2 = "); rdd_out_str(ddnorm_av[1]);  printf("\n");
	printf("||DDA_sp^T * V||_2 = "); rdd_out_str(ddnorm_atv[0]); printf("\n");
	printf("||DDA^T    * V||_2 = "); rdd_out_str(ddnorm_atv[1]); printf("\n");

	//goto QD;
// -------------
// Triple-double
// -------------
TD:
    // MPFRSMatrix -> TDRSMatrix
	printf("td!\n");
    mpfa_sp = init2_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
    printf("row_dim, nzero_total_num = %ld, %ld\n", mpfa_sp->row_dim, mpfa_sp->nzero_total_num);
    tda_sp = init_set_tdrsmatrix_mpfrsmatrix(mpfa_sp);
    //print_tdrsmatrix(tda_sp);
	// A * v
	tdvec_x = init_tdvector(tda_sp->col_dim);
	tdvec_b = init_tdvector(tda_sp->row_dim);

	get_tdvector(tdvec_x);
	print_tdvector(tdvec_x);
	printf("mul!"); fflush(stdout);
	mul_tdrsmatrix_tdvec(tdvec_b, tda_sp, tdvec_x);
	printf("mul_done!");fflush(stdout);
	norm2_tdvector(tdnorm_av[0], tdvec_b);
	print_tdvector(tdvec_b);

	get_tdvector(tdvec_b);
	mul_tdrsmatrixt_tdvec(tdvec_x, tda_sp, tdvec_b);
	printf("tdvec_x := \n"); print_tdvector(tdvec_x);
	norm2_tdvector(tdnorm_atv[0], tdvec_x);
	//printf("||TDA^T    * V||_2 = "); rtd_out_str(tdnorm_atv[0]); printf("\n");
	print_tdvector(tdvec_x);

	free_tdvector(tdvec_x);
	free_tdvector(tdvec_b);
    free_tdrsmatrix(tda_sp);
    free_mpfrsmatrix(mpfa_sp);

	// Read MM file as DENSE matrix
    printf("Reading %s...\n", MATRIX_MM_FILE);
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	print_mpfmatrix(mpfa);
	tda = init_tdmatrix(mpfa->row_dim, mpfa->col_dim);
	subst_tdmatrix_mpfmat(tda, mpfa);
    //print_tdmatrix(tda);

	// A * v
	tdvec_x = init_tdvector(tda->col_dim);
	tdvec_b = init_tdvector(tda->row_dim);

	get_tdvector(tdvec_x);
	mul_tdmatrix_tdvec(tdvec_b, tda, tdvec_x);
	norm2_tdvector(tdnorm_av[1], tdvec_b);
	print_tdvector(tdvec_b);

	get_tdvector(tdvec_b);
	mul_tdmatrixt_tdvec(tdvec_x, tda, tdvec_b);
	norm2_tdvector(tdnorm_atv[1], tdvec_x);
	print_tdvector(tdvec_x);

	free_tdvector(tdvec_x);
	free_tdvector(tdvec_b);
	free_tdmatrix(tda);
	free_mpfmatrix(mpfa);

	// print norm
	printf("||TDA_sp   * V||_2 = "); rtd_out_str(tdnorm_av[0]);  printf("\n");
	printf("||TDA      * V||_2 = "); rtd_out_str(tdnorm_av[1]);  printf("\n");
	printf("||TDA_sp^T * V||_2 = "); rtd_out_str(tdnorm_atv[0]); printf("\n");
	printf("||TDA^T    * V||_2 = "); rtd_out_str(tdnorm_atv[1]); printf("\n");

// -------------
// Quad-double
// -------------
QD:
    // MPFRSMatrix -> QDRSMatrix
    mpfa_sp = init_mpfrsmatrix_readMMcoordinate(MATRIX_MM_FILE);
    printf("row_dim, nzero_total_num = %ld, %ld\n", mpfa_sp->row_dim, mpfa_sp->nzero_total_num);
    qda_sp = init_set_qdrsmatrix_mpfrsmatrix(mpfa_sp);
    //print_qdrsmatrix(qda_sp);
	// A * v
	qdvec_x = init_qdvector(qda_sp->col_dim);
	qdvec_b = init_qdvector(qda_sp->row_dim);

	get_qdvector(qdvec_x);
	mul_qdrsmatrix_qdvec(qdvec_b, qda_sp, qdvec_x);
	norm2_qdvector(qdnorm_av[0], qdvec_b);
	print_qdvector(qdvec_b);

	get_qdvector(qdvec_b);
	mul_qdrsmatrixt_qdvec(qdvec_x, qda_sp, qdvec_b);
	norm2_qdvector(qdnorm_atv[0], qdvec_x);
	print_qdvector(qdvec_x);

	//printf("||QDA_sp   * V||_2 = "); rqd_out_str(qdnorm_av[0]);  printf("\n");
	//printf("||QDA_sp^T * V||_2 = "); rqd_out_str(qdnorm_atv[0]); printf("\n");

	free_qdvector(qdvec_x);
	free_qdvector(qdvec_b);
    free_qdrsmatrix(qda_sp);
    free_mpfrsmatrix(mpfa_sp);

	// Read MM file as DENSE matrix
    printf("Reading %s...\n", MATRIX_MM_FILE);
	mpfa = init2_mpfmatrix_readMMcoordinate(MATRIX_MM_FILE, PREC);
	qda = init_qdmatrix(mpfa->row_dim, mpfa->col_dim);
	subst_qdmatrix_mpfmat(qda, mpfa);
    //print_qdmatrix(qda);

	// A * v
	qdvec_x = init_qdvector(qda->col_dim);
	qdvec_b = init_qdvector(qda->row_dim);

	get_qdvector(qdvec_x);
	mul_qdmatrix_qdvec(qdvec_b, qda, qdvec_x);
	norm2_qdvector(qdnorm_av[1], qdvec_b);
	print_qdvector(qdvec_b);

	get_qdvector(qdvec_b);
	mul_qdmatrixt_qdvec(qdvec_x, qda, qdvec_b);
	norm2_qdvector(qdnorm_atv[1], qdvec_x);
	print_qdvector(qdvec_x);

	free_qdvector(qdvec_x);
	free_qdvector(qdvec_b);
	free_qdmatrix(qda);
	free_mpfmatrix(mpfa);

	// print norm
	printf("||QDA_sp   * V||_2 = "); rqd_out_str(qdnorm_av[0]);  printf("\n");
	printf("||QDA      * V||_2 = "); rqd_out_str(qdnorm_av[1]);  printf("\n");
	printf("||QDA_sp^T * V||_2 = "); rqd_out_str(qdnorm_atv[0]); printf("\n");
	printf("||QDA^T    * V||_2 = "); rqd_out_str(qdnorm_atv[1]); printf("\n");



#endif // USE_GMP
}


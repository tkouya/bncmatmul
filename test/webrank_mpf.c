/********************************************************************************/
/*                                                                              */
/* webrank_mpf.c : Getting WebRank for numerous Web pages                       */
/* Copyright (c) 2011 Tomonori Kouya, All rights reserved.                      */
/*                                                                              */
/* Usage    : % webrank_mpfr urimatrix.dat > webrank.txt                        */
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

#include "bncsparse.h"

/* show usage of webrank */
void usage(const char *progname)
{
	printf("Usage: %s [uri_matrix.dat]\n", progname);
}

/* Print webrank based on DVector */
#define WEBRANK_DIV 10

typedef struct {
//	double element;
	mpf_t element;
	long int index;
} webrank_element;

typedef struct {
	webrank_element *element;
	long int dim;
} webrank;

typedef webrank *Webrank;

/* Initialize DVector */
Webrank init_webrank(long int dim)
{
	Webrank ret;

	if(dim < 0)
		return NULL;

	ret = (Webrank)malloc(sizeof(webrank));
	if(ret == NULL)
		return NULL;

	ret->dim = dim;
//	ret->element = (webrank_element *)calloc((size_t)dim, sizeof(double) + sizeof(long int));
	ret->element = (webrank_element *)calloc((size_t)dim, sizeof(mpf_t) + sizeof(long int));
	if(ret->element == NULL)
		return NULL;

	return ret;
}

/* Clean DVector */
void free_webrank(Webrank rank)
{
	free(rank->element);
	free(rank);
}

void subst_webrank_mpfvec(Webrank rank, MPFVector vec)
{
	long int i;

	for(i = 0; i < rank->dim; i++)
	{
//		(rank->element[i]).element = vec->element[i];
		mpf_set((rank->element[i]).element, get_mpfvector_i(vec, i));
		(rank->element[i]).index = i;
	}
}

int compare_webrank(const webrank_element *val1, const webrank_element *val2)
{
//	double dval1, dval2;

//	printf("%e(%ld) %e(%ld) \n", val1->element, val1->index, val2->element, val2->index);
	if(mpf_cmp(val1->element, val2->element) < 0)
		return 1; // reverse order
	else if(mpf_cmp(val1->element, val2->element) == 0)
		return 0;
	else
		return -1;
}

/* Main function */
int main(int argc, char *argv[])
{
	MPFRSMatrix webrank_mat = NULL;
	MPFVector webrank_vec = NULL;
	long int row_dim, *nzero_col_dim, nzero_total_num, i;
	mpf_t max_eig, reps, aeps;
	double stime, etime;

	/* if no options, print usage */
	if(argc <= 1)
	{
		usage(argv[0]);
		return EXIT_SUCCESS;
	}

	set_bnc_default_prec(128);
	mpf_init(max_eig);
	mpf_init(reps); mpf_set_d(reps, 1.0e-5);
	mpf_init(aeps); mpf_set_d(aeps, 1.0e-50);

	/* Read URI link data */
	if(get_vars_mpfrsmatrix_fname(&row_dim, &nzero_col_dim, &nzero_total_num, argv[1]) == ERROR)
		return EXIT_FAILURE;
	fprintf(stderr, "row_dim = %ld, nzero_total_num = %ld\n", row_dim, nzero_total_num);
	for(i = 0; i < row_dim; i++)
		fprintf(stderr, "nzero_col_dim[%ld] = %ld\n", i, nzero_col_dim[i]);

	webrank_mat = init_mpfrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	if(webrank_mat == NULL)
		return EXIT_FAILURE;

	if(fread_urilinkdat_fname_mpf(webrank_mat, argv[1]) == ERROR)
		return EXIT_FAILURE;

	/* Get webrank by Power method */
	fprintf(stderr, "Webrank: %ld URIs\n", webrank_mat->col_dim);

	webrank_vec = init_mpfvector(webrank_mat->col_dim);
	stime = get_secv();
	//max_eig = dpower_rsmatrix(webrank_vec, webrank_mat, 1.0e-5, 1.0e-50, 100);
	//max_eig = dpower_rsmatrix_G(webrank_vec, webrank_mat, 0.9, 1.0e-5, 1.0e-50, 100);
	mpfpower_rsmatrix(max_eig, webrank_vec, webrank_mat, reps, aeps, 100);
	etime = get_secv();
	print_mpfvector(webrank_vec);
//	print_webrank_dvector(webrank_vec);
	fprintf(stderr, "max_eig      : %f\n", mpf_get_d(max_eig));
	fprintf(stderr, "CPU time(sec): %f\n", etime - stime);

	/* free webrank_* */
	free_mpfrsmatrix(webrank_mat);
	free_mpfvector(webrank_vec);
	mpf_clear(max_eig);
	mpf_clear(reps);
	mpf_clear(aeps);

	return EXIT_SUCCESS;
}

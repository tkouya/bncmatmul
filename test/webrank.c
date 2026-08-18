/********************************************************************************/
/*                                                                              */
/* webrank.c : Getting WebRank for numerous Web pages                           */
/* Copyright (c) 2006-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* Version 0.1: 2007-06-16: build the first version                             */
/* Version 0.2: 2007-11-13: append dpower_rsmatrix_G                            */
/* Version 0.3: 2011-06-28: change the way of compiling                         */
/*                                                                              */
/* Usage    : % webrank urimatrix.dat > webrank.txt                             */
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
/*                                                         */
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
	double element;
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
	ret->element = (webrank_element *)calloc((size_t)dim, sizeof(double) + sizeof(long int));
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

void subst_webrank_dvec(Webrank rank, DVector vec)
{
	long int i;

	for(i = 0; i < rank->dim; i++)
	{
		(rank->element[i]).element = vec->element[i];
		(rank->element[i]).index = i;
	}
}

int compare_webrank(const webrank_element *val1, const webrank_element *val2)
{
	double dval1, dval2;

	dval1 = val1->element;
	dval2 = val2->element;
//	printf("%e(%ld) %e(%ld) \n", val1->element, val1->index, val2->element, val2->index);
	if(dval1 < dval2)
		return 1; // reverse order
	else if(dval1 == dval2)
		return 0;
	else
		return -1;
}

void print_webrank_dvector(DVector vec)
{
	long int i, j, threshold_num[WEBRANK_DIV], index, rank_num;
	double max_val, min_val, val, threshold[WEBRANK_DIV + 1], rank;

	Webrank webrank_tmp;

	webrank_tmp = init_webrank(vec->dim);
	subst_webrank_dvec(webrank_tmp, vec);

	/* get max and min vals */
	max_val = vec->element[0];
	min_val = vec->element[0];
	for(i = 1; i < vec->dim; i++)
	{
		val = vec->element[i];
		if(max_val < val)
			max_val = val;
		if(min_val > val)
			min_val = val;
	}

	/* threshhold of webrank */
//	for(i = 0; i <= WEBRANK_DIV; i++)
//		threshold[i] = min_val + (double)i * (max_val - min_val) / WEBRANK_DIV;

	// sort in reverse order
	qsort(webrank_tmp->element, webrank_tmp->dim, sizeof(webrank_element), (int (*)(const void *, const void *))compare_webrank);

	for(i = 0; i < vec->dim; i++)
	{
		/* get rank */
		index = (webrank_tmp->element[i]).index;
		rank = (double) (vec->dim - i - 1) / (double)(vec->dim - 1);
		vec->element[index] = rank; // substitution
	}

	for(i = 0; i < vec->dim; i++)
	{
//		rank = min_val + (vec->element[i] - min_val) * / max_val;
//		fprintf(stderr, "%10d, %f\n", i, vec->element[i]);
//		fprintf(stderr, "%2d %2d, %f\n", i, (webrank_tmp->element[i]).index, (webrank_tmp->element[i]).element);
		printf("%10ld, %f\n", i, vec->element[i]);
	}

//	for(i = 0; i < WEBRANK_DIV; i++)
//		fprintf(stderr, "%10d: %e -- %e\n", i + 1, threshold[i], threshold[i + 1]);
	fprintf(stderr, "max_raw_element: %g, min_raw_element: %g\n", max_val, min_val);

	free_webrank(webrank_tmp);
}

/* Power Method for G = alpha * H + (1/n)*e * (alpha * e_i^T + (1-alpha)*e^T) */
/* 	double *evec: the eigenvector for max eigenvalue */
/* 	double *drsmat (H) :  Randomly sparse matrix */
/* double alpha: 0 < alpha < 1 */
/* 	double reps, aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
double dpower_rsmatrix_G(DVector evec, DRSMatrix mat, double alpha, double reps, double aeps, long int max_times)
{
	long int dim, i, absmax_index, times;
	double absmax_new_evec, max_eig, old_max_eig, coef;
	DVector new_evec, tmp_evec;

	/* n = dim */
	dim = mat->row_dim;

	new_evec = init_dvector(mat->row_dim);
	tmp_evec = init_dvector(mat->row_dim);

	/* initialize evec */
	for(i = 0; i < evec->dim; i++)
		evec->element[i] = 1.0;

	/* set nzero_row_dim */
	set_nzero_row_dim(mat);

	/* main loop */
	old_max_eig = 0.0;
	for(times = 0; times < max_times; times++)
	{
		/* w := alpha * H * x */
		mul_drsmatrix_dvec(new_evec, mat, evec);
		smul_dvector(new_evec, alpha, new_evec);

		/* coef := (alpha * x_i + (1 - alph) * sum^n_{j=1} x_j) / n */
		coef = 0.0;
		for(i = 0; i < dim; i++)
			coef += get_dvector_i(evec, i);
		coef *= (1.0 - alpha);

		for(i = 0; i < dim; i++)
		{
			if(mat->nzero_row_dim[i] == 0)
				coef += alpha * get_dvector_i(evec, i);
		}
		coef /= (double)dim;

		/* coef * e */
		for(i = 0; i < dim; i++)
			set_dvector_i(tmp_evec, i, coef);

		/* w := alpha * H * x + coef * x */
		add_dvector(new_evec, new_evec, tmp_evec);

		absmax_index = absmax_index_dvector(&absmax_new_evec, new_evec);
		max_eig = absmax_new_evec / evec->element[absmax_index];
//		smul_dvector(evec, 1.0 / absmax_new_evec, new_evec);
		smul_dvector(evec, 1.0 / norm1_dvector(new_evec), new_evec); // Baba's example
		if((fabs(max_eig - old_max_eig) <= reps * fabs(old_max_eig) + aeps) && (times >= 2))
		{
			fprintf(stderr, "Convergent!(Iterative Times = %ld)\n", times);
			break;
		}
		if(times % 10 == 0)
			fprintf(stderr, "%5ld %25.17e\n", times, max_eig);
		old_max_eig = max_eig;
	}

	/* free */
	free_dvector(tmp_evec);
	free_dvector(new_evec);

	return max_eig;
}

/* Main function */
int main(int argc, char *argv[])
{
	DRSMatrix webrank_mat = NULL;
	DVector webrank_vec = NULL;
	long int row_dim, *nzero_col_dim, nzero_total_num, i;
	double max_eig;
	double stime, etime;

	/* if no options, print usage */
	if(argc <= 1)
	{
		usage(argv[0]);
		return EXIT_SUCCESS;
	}

	/* Read URI link data */
	if(get_vars_drsmatrix_fname(&row_dim, &nzero_col_dim, &nzero_total_num, argv[1]) == ERROR)
		return EXIT_FAILURE;
	fprintf(stderr, "row_dim = %ld, nzero_total_num = %ld\n", row_dim, nzero_total_num);
	for(i = 0; i < row_dim; i++)
		fprintf(stderr, "nzero_col_dim[%ld] = %ld\n", i, nzero_col_dim[i]);

	webrank_mat = init_drsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	if(webrank_mat == NULL)
		return EXIT_FAILURE;

	if(fread_urilinkdat_fname(webrank_mat, argv[1]) == ERROR)
		return EXIT_FAILURE;

	/* Get webrank by Power method */
	fprintf(stderr, "Webrank: %ld URIs\n", webrank_mat->col_dim);

	webrank_vec = init_dvector(webrank_mat->col_dim);
	stime = get_secv();
	//max_eig = dpower_rsmatrix(webrank_vec, webrank_mat, 1.0e-5, 1.0e-50, 100);
	max_eig = dpower_rsmatrix_G(webrank_vec, webrank_mat, 0.9, 1.0e-5, 1.0e-50, 100);
	etime = get_secv();
	print_dvector(webrank_vec);
//	print_webrank_dvector(webrank_vec);
	fprintf(stderr, "max_eig      : %f\n", max_eig);
	fprintf(stderr, "CPU time(sec): %f\n", etime - stime);

	/* free webrank_* */
	free_drsmatrix(webrank_mat);
	free_dvector(webrank_vec);

	return EXIT_SUCCESS;
}

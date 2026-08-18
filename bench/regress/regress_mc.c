/* regress_mc.c - numerical regression check for multicomponent complex
 * precisions (cdd, ctd, cqd, cds, cts, cqs).
 *
 * Deterministic inputs; prints per-run:
 *   REG,<prec>,<backend>,<dim>,<mm_relerr>,<mv_relerr>,<lu_relerr>,<mm_digest>
 *
 *   mm_relerr / mv_relerr : max relative error of mul_*matrix / matvec
 *                           vs a double-complex reference (structural check)
 *   lu_relerr             : limb-wise max relative error of LUdecompPM+Solve
 *                           against the known solution (full-precision check)
 *   mm_digest             : weighted limb sum of the matmul result, printed
 *                           with %.17e (bit-level change detector)
 *
 * Build with: -DP=cdd -DPU=CDD -DMT=CDDMatrix -DVT=CDDVector -DPSIZE=DDSIZE
 *             [-DSINGLE_BASED] -DPREC_NAME=\"cdd\" -DBACKEND_NAME=\"serial\"
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#ifdef SINGLE_BASED
#include "cdslinear.h"
#include "ctslinear.h"
#include "cqslinear.h"
#else
#include "cddlinear.h"
#include "ctdlinear.h"
#include "cqdlinear.h"
#endif

#define _C(a,b) a##b
#define C(a,b) _C(a,b)
#define FN(pre,post) C(C(pre,P),post)
#define UFN(post) C(PU,post)
#define ELEMT C(P,float)
#define MATVEC C(C(C(C(mul_,P),matrix_),P),vec)
#define GETM C(C(C(get_,P),matrix_ij_),ELEMT)
#define GETV C(C(C(get_,P),vector_i_),ELEMT)

static double _Complex e2dc(ELEMT e)
{
	double re = 0.0, im = 0.0;
	int k;
	for(k = PSIZE - 1; k >= 0; k--){
		re += (double)e.val_re[k];
		im += (double)e.val_im[k];
	}
	return re + im * I;
}

int main(int argc, char **argv)
{
	long dim = (argc >= 2) ? atol(argv[1]) : 96;
	long i, j, k;
	int l;
	double mm_relerr = 0.0, mv_relerr = 0.0, lu_relerr = 0.0, digest = 0.0;

	MT A = FN(init_,matrix)(dim, dim), B = FN(init_,matrix)(dim, dim);
	MT Cm = FN(init_,matrix)(dim, dim), LU = FN(init_,matrix)(dim, dim);
	VT x = FN(init_,vector)(dim), b = FN(init_,vector)(dim), y = FN(init_,vector)(dim);
	double _Complex *Ad = calloc(dim * dim, sizeof(double _Complex));
	double _Complex *Bd = calloc(dim * dim, sizeof(double _Complex));
	double _Complex *xd = calloc(dim, sizeof(double _Complex));
	long *ch = calloc(dim + 1, sizeof(long));

	for(i = 0; i < dim; i++){
		for(j = 0; j < dim; j++){
			double _Complex av = (((i*131 + j*17) % 97) * 0.01 + 0.1)
			                   + (((i*3 + j*5) % 41) * 0.01 + 0.05) * I;
			double _Complex bv = (((i*13 + j*131) % 89) * 0.01 + 0.1)
			                   + (((i*7 + j*11) % 37) * 0.01 + 0.05) * I;
			FN(set_,matrix_ij_cd)(A, i, j, av);
			FN(set_,matrix_ij_cd)(LU, i, j, av);
			FN(set_,matrix_ij_cd)(B, i, j, bv);
			Ad[i * dim + j] = av;
			Bd[i * dim + j] = bv;
		}
		/* diagonal dominance so LU without luck-dependent pivoting stays stable */
		{
			double _Complex dv = (double)dim + ((i % 7) * 0.125 + 1.0)
			                   + ((i % 5) * 0.0625 + 0.5) * I;
			FN(set_,matrix_ij_cd)(A, i, i, dv);
			FN(set_,matrix_ij_cd)(LU, i, i, dv);
			Ad[i * dim + i] = dv;
		}
		{
			double _Complex xv = (((i*29 + 5) % 83) * 0.01 + 0.1)
			                   + (((i*2 + 1) % 31) * 0.01 + 0.05) * I;
			FN(set_,vector_i_cd)(x, i, xv);
			xd[i] = xv;
		}
	}

	/* matmul vs double-complex reference + bit digest */
	FN(mul_,matrix)(Cm, A, B);
	for(i = 0; i < dim; i++){
		for(j = 0; j < dim; j++){
			double _Complex ref = 0.0;
			double rel;
			ELEMT e;
			for(k = 0; k < dim; k++)
				ref += Ad[i * dim + k] * Bd[k * dim + j];
			e = GETM(Cm, i, j);
			rel = cabs(e2dc(e) - ref) / cabs(ref);
			if(rel > mm_relerr) mm_relerr = rel;
			for(l = 0; l < PSIZE; l++)
				digest += ((double)e.val_re[l] + 2.0 * (double)e.val_im[l])
				        * (double)((i * 31 + j * 17 + l * 7) % 13 + 1);
		}
	}

	/* matvec vs double-complex reference */
	MATVEC(y, A, x);
	for(i = 0; i < dim; i++){
		double _Complex ref = 0.0;
		double rel;
		for(k = 0; k < dim; k++)
			ref += Ad[i * dim + k] * xd[k];
		rel = cabs(e2dc(GETV(y, i)) - ref) / cabs(ref);
		if(rel > mv_relerr) mv_relerr = rel;
	}

	/* LU solve: b = A*x (library matvec), solve, limb-wise compare vs x */
	MATVEC(b, A, x);
	if(UFN(LUdecompPM)(LU, ch) != 0){ fprintf(stderr, "LUdecompPM failed\n"); return 1; }
	if(C(C(Solve,PU),LSPM)(y, LU, b, ch) != 0){ fprintf(stderr, "Solve failed\n"); return 1; }
	for(i = 0; i < dim; i++){
		ELEMT ye = GETV(y, i), xe = GETV(x, i);
		double den = fmax(fabs((double)xe.val_re[0]), fabs((double)xe.val_im[0]));
		double num = 0.0;
		for(l = 0; l < PSIZE; l++){
			num = fmax(num, fabs((double)ye.val_re[l] - (double)xe.val_re[l]));
			num = fmax(num, fabs((double)ye.val_im[l] - (double)xe.val_im[l]));
		}
		if(num / den > lu_relerr) lu_relerr = num / den;
	}

	printf("REG,%s,%s,%ld,%.6e,%.6e,%.6e,%.17e\n",
	       PREC_NAME, BACKEND_NAME, dim, mm_relerr, mv_relerr, lu_relerr, digest);

	FN(free_,matrix)(A); FN(free_,matrix)(B); FN(free_,matrix)(Cm); FN(free_,matrix)(LU);
	FN(free_,vector)(x); FN(free_,vector)(b); FN(free_,vector)(y);
	free(Ad); free(Bd); free(xd); free(ch);
	return 0;
}

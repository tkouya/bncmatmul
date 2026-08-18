/* lu_illcond_mr.c - ill-conditioned test cases (Hilbert / Frank matrices) for
 * the FMA+SIMD LU decomposition and forward/backward substitution.
 *
 * The matrix entries are stored base-word only (double resp. float), so the
 * SAME matrix is factorized by the nofma and fma binaries of one precision.
 * The reference right-hand side b = A * ones is accumulated in QD arithmetic
 * (rqd_add/rqd_mul only -- independent of BNC_USE_NEW_FMA) from the STORED
 * entries and then truncated to the working precision; the forward error of
 * the computed solution against x = ones is also evaluated in QD arithmetic
 * so that errors far below double precision remain measurable.
 *
 * Build with: -DP=<dd|..> -DPU=<DD|..> -DMT=.. -DVT=.. -DBASE=<double|float>
 *             -DPSIZE=.. -DBASE_IS_DOUBLE=<1|0> -DPREC_NAME=\"dd\"
 *             -DBACKEND_NAME=\"serial\" [-DBNC_USE_NEW_FMA]
 * Usage: ./lu_ill_<p>_<bk>_<fm> <family:hilbert|frank> <dim>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ddlinear.h"
#include "tdlinear.h"
#include "qdlinear.h"
#include "dslinear.h"
#include "tslinear.h"
#include "qslinear.h"

#define _C(a,b) a##b
#define C(a,b) _C(a,b)
#define FN(pre,post) C(C(pre,P),post)
#define UFN(post) C(PU,post)

#ifndef PREC_NAME
#define PREC_NAME "mr"
#endif
#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif
#ifdef BNC_USE_NEW_FMA
#define FMA_NAME "fma"
#else
#define FMA_NAME "nofma"
#endif

static void mkval(BASE *v, double x){ v[0]=(BASE)x; for(int k=1;k<PSIZE;k++) v[k]=(BASE)0; }

/* q += d, in QD arithmetic (flag-independent) */
static void qd_add_d(double q[4], double d){
    double t[4] = {d, 0.0, 0.0, 0.0};
    rqd_add(q, q, t);
}

/* truncate a QD value to the working precision components */
static void qd_to_target(BASE out[PSIZE], const double q[4]){
#if BASE_IS_DOUBLE
    for(int c = 0; c < PSIZE; c++) out[c] = q[c];
#else
    double qq[4], t[4];
    memcpy(qq, (const void *)q, sizeof qq);
    for(int c = 0; c < PSIZE; c++){
        float f = (float)qq[0];
        out[c] = f;
        t[0] = (double)f; t[1] = t[2] = t[3] = 0.0;
        rqd_sub(qq, qq, t);
    }
#endif
}

int main(int argc, char**argv){
    const char *family = (argc>=2)? argv[1] : "hilbert";
    long dim = (argc>=3)? atol(argv[2]) : 8;
    long i, j;
    BASE tmp[PSIZE];
    int hilbert = (family[0]=='h');

    MT A  = FN(init_,matrix)(dim,dim);
    MT LU = FN(init_,matrix)(dim,dim);
    VT b = FN(init_,vector)(dim), y = FN(init_,vector)(dim);
    long *ch = (long *)calloc((size_t)dim, sizeof(long));

    /* stored entries, identical for the nofma/fma binaries of one precision:
     * hilbert: 1/(i+j+1) computed in QD (c_qd_div, flag-independent) and
     *          truncated to the full working precision;
     * frank:   exact small integers. */
    for(i=0;i<dim;i++){
        for(j=0;j<dim;j++){
            if(hilbert){
                double one[4] = {1.0, 0.0, 0.0, 0.0};
                double den[4] = {(double)(i + j + 1), 0.0, 0.0, 0.0};
                double q[4];
                rqd_div(q, one, den);
                qd_to_target(tmp, q);
            }
            else{ /* Frank (upper Hessenberg, MATLAB gallery('frank')) */
                double v;
                if(i <= j)          v = (double)(dim - j);
                else if(i == j + 1) v = (double)(dim - j - 1);
                else                v = 0.0;
                mkval(tmp, v);
            }
            FN(set_,matrix_ij)(A,i,j,tmp);
        }
    }

    /* reference rhs b = A * ones, accumulated in QD from the stored entries */
    for(i=0;i<dim;i++){
        double q[4] = {0.0, 0.0, 0.0, 0.0};
        for(j=0;j<dim;j++){
            memcpy(tmp, FN(get_,matrix_ij)(A, i, j), sizeof(BASE) * PSIZE);
            for(int c = 0; c < PSIZE; c++) qd_add_d(q, (double)tmp[c]);
        }
        qd_to_target(tmp, q);
        FN(set_,vector_i)(b,i,tmp);
    }

    /* factorize + solve */
    {
        long n = A->real_row_dim * A->real_col_dim;
        for(int c = 0; c < PSIZE; c++)
            memcpy(LU->element[c], A->element[c], (size_t)n * sizeof(BASE));
    }
    if(UFN(LUdecompPM)(LU, ch) != 0){ fprintf(stderr, "decomp failed\n"); return 1; }
    if(C(Solve,C(PU,LSPM))(y, LU, b, ch) != 0){ fprintf(stderr, "solve failed\n"); return 1; }

    /* forward error vs x = ones, evaluated in QD */
    double maxrelerr = 0.0;
    for(i=0;i<dim;i++){
        double q[4] = {0.0, 0.0, 0.0, 0.0}, e;
        memcpy(tmp, FN(get_,vector_i)(y, i), sizeof(BASE) * PSIZE);
        for(int c = 0; c < PSIZE; c++) qd_add_d(q, (double)tmp[c]);
        qd_add_d(q, -1.0);
        e = fabs(q[0]);
        if(e > maxrelerr) maxrelerr = e;
    }

    printf("ILL,%s,%s,%s,%s,%ld,%.6e\n",
           PREC_NAME, BACKEND_NAME, FMA_NAME, hilbert ? "hilbert" : "frank",
           dim, maxrelerr);

    FN(free_,matrix)(A); FN(free_,matrix)(LU);
    FN(free_,vector)(b); FN(free_,vector)(y);
    free(ch);
    return 0;
}

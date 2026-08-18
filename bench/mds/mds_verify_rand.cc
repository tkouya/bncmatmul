/* mds_verify_rand.cc : random-operand cross-backend check for the EFT types.
   The companion to mds_verify.cc, which uses small structured values; here
   every entry is a pseudo-random number in [-1,1) from a fixed xorshift
   stream, so all builds see identical inputs.  Each line prints a hash over
   every limb (so serial/avx2/avx512 can be diffed byte-for-byte) plus a plain
   double sum (so the size of any difference is visible).  Covers axpy
   (add_cmul), matvec and matmul for dd/td/qd and ds/ts/qs.

     g++ -O3 -ffp-contract=off [-mavx2 -mfma | -mavx512f -mfma] -Iinclude \
         bench/mds/mds_verify_rand.cc -L. -lbncmatmul-0.24[_avx2|_avx512] ...

   Element-wise kernels must match exactly across backends; GEMV/GEMM may
   differ in the last limb for dd and ds because the lane-wise accumulation
   reassociates the dot product (see Sec. 1.3 of the User's Guide).        */
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#define USE_DDLINEAR
#define USE_TDLINEAR
#define USE_QDLINEAR
#define USE_DSLINEAR
#define USE_TSLINEAR
#define USE_QSLINEAR
#include "matmul_strassen.h"
#include <qd/qd_real.h>

static uint64_t H;
static double SUM;
static void mix(double v){ uint64_t u; memcpy(&u,&v,8); H ^= u + 0x9e3779b97f4a7c15ULL + (H<<6) + (H>>2); SUM += v; }
static void mixf(float v){ uint32_t u; memcpy(&u,&v,4); H ^= (uint64_t)u + 0x9e3779b97f4a7c15ULL + (H<<6) + (H>>2); SUM += (double)v; }

/* xorshift so both builds see identical inputs */
static uint64_t st = 88172645463325252ULL;
static double urand(void){ st^=st<<13; st^=st>>7; st^=st<<17; return (double)(st>>11)/9007199254740992.0*2.0-1.0; }

#define DEFVER(TOK, NAME, BASE, SZ, MIXF)                                     \
static void ver_##TOK(long n)                                                 \
{                                                                             \
    st = 88172645463325252ULL;                                                \
    TOK##matrix *A=init_##TOK##matrix(n,n),*B=init_##TOK##matrix(n,n),*C=init_##TOK##matrix(n,n); \
    TOK##vector *x=init_##TOK##vector(n),*y=init_##TOK##vector(n),*z=init_##TOK##vector(n); \
    BASE t[SZ];                                                               \
    for(long i=0;i<n;i++){                                                    \
        for(long j=0;j<n;j++){                                                \
            for(int k=0;k<SZ;k++) t[k]=(BASE)0; t[0]=(BASE)urand();           \
            set_##TOK##matrix_ij(A,i,j,t);                                    \
            for(int k=0;k<SZ;k++) t[k]=(BASE)0; t[0]=(BASE)urand();           \
            set_##TOK##matrix_ij(B,i,j,t);                                    \
        }                                                                     \
        for(int k=0;k<SZ;k++) t[k]=(BASE)0; t[0]=(BASE)urand();               \
        set_##TOK##vector_i(x,i,t);                                           \
        for(int k=0;k<SZ;k++) t[k]=(BASE)0; t[0]=(BASE)urand();               \
        set_##TOK##vector_i(y,i,t);                                           \
    }                                                                         \
    BASE al[SZ]; for(int k=0;k<SZ;k++) al[k]=(BASE)0; al[0]=(BASE)urand();     \
    add_cmul_##TOK##vector(z,y,al,x);                                         \
    H=0; SUM=0; for(long i=0;i<n;i++) for(int k=0;k<SZ;k++) MIXF(get_##TOK##vector_i(z,i)[k]); \
    printf("%-3s axpy   %016llx %.17e\n", NAME, (unsigned long long)H, SUM);              \
    mul_##TOK##matrix_##TOK##vec(y,A,x);                                      \
    H=0; SUM=0; for(long i=0;i<n;i++) for(int k=0;k<SZ;k++) MIXF(get_##TOK##vector_i(y,i)[k]); \
    printf("%-3s matvec %016llx %.17e\n", NAME, (unsigned long long)H, SUM);              \
    mul_##TOK##matrix(C,A,B);                                                 \
    H=0; SUM=0; for(long i=0;i<n;i++) for(long j=0;j<n;j++) for(int k=0;k<SZ;k++) MIXF(get_##TOK##matrix_ij(C,i,j)[k]); \
    printf("%-3s matmul %016llx %.17e\n", NAME, (unsigned long long)H, SUM);              \
}
DEFVER(dd,"dd",double,DDSIZE,mix)
DEFVER(td,"td",double,TDSIZE,mix)
DEFVER(qd,"qd",double,QDSIZE,mix)
DEFVER(ds,"ds",float, DSSIZE,mixf)
DEFVER(ts,"ts",float, TSSIZE,mixf)
DEFVER(qs,"qs",float, QSSIZE,mixf)

int main(int argc,char**argv){
    unsigned int cw; fpu_fix_start(&cw);
    long n = (argc>=2)?atol(argv[1]):200;
    ver_dd(n); ver_td(n); ver_qd(n); ver_ds(n); ver_ts(n); ver_qs(n);
    return 0;
}

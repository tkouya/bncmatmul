/* mm_float.c - native float matmul benchmark, SELF-CONTAINED.
 *
 * The library ships only a triple-loop float multiply (mul_fmatrix), so the
 * blocking / Strassen / Winograd algorithms are implemented here directly on
 * row-major float arrays.  SIMD comes from the compiler (-mcpu + auto-vec):
 * scalar (-O3) vs NEON (cortex-a76) vs SVE2 (neoverse-v2), matching the
 * cf self-contained benches in this tree.
 *
 * Modes/output identical to mm_multi.c / mm_double.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpfr.h>

#ifndef PREC_NAME
#define PREC_NAME "float"
#endif
#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif
#ifndef MIN_DIM
#define MIN_DIM 32
#endif
#ifndef REF_BITS
#define REF_BITS 512
#endif

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }
#define TIME_OP(tvar, CALL) do { long it=1; double e; for(;;){ \
    double s=now_s(); for(long _k=0;_k<it;_k++){ CALL; } e=now_s()-s; \
    if(e>=0.20 || it>=(1L<<22)) break; it*=2; } tvar=e/(double)it; } while(0)

/* ---- algorithms on row-major float, leading dimension = ld ---- */

/* C = A*B, n x n, base triple loop (i-k-j for cache/vectorization) */
static void triple(float *C,long ldc,const float *A,long lda,const float *B,long ldb,long n){
    for(long i=0;i<n;i++){
        for(long j=0;j<n;j++) C[i*ldc+j]=0.0f;
        for(long k=0;k<n;k++){
            float a=A[i*lda+k]; const float *br=B+k*ldb; float *cr=C+i*ldc;
            for(long j=0;j<n;j++) cr[j]+=a*br[j];
        }
    }
}

/* blocked (tiled) multiply, tile = bs */
static void block(float *C,long ldc,const float *A,long lda,const float *B,long ldb,long n,long bs){
    for(long i=0;i<n;i++) for(long j=0;j<n;j++) C[i*ldc+j]=0.0f;
    for(long ii=0;ii<n;ii+=bs) for(long kk=0;kk<n;kk+=bs) for(long jj=0;jj<n;jj+=bs){
        long iM=ii+bs<n?ii+bs:n, kM=kk+bs<n?kk+bs:n, jM=jj+bs<n?jj+bs:n;
        for(long i=ii;i<iM;i++) for(long k=kk;k<kM;k++){
            float a=A[i*lda+k]; const float *br=B+k*ldb; float *cr=C+i*ldc;
            for(long j=jj;j<jM;j++) cr[j]+=a*br[j];
        }
    }
}

/* dst = a (+/-) b over h x h */
static void madd(float *d,long ldd,const float *a,long lda,const float *b,long ldb,long h,int sub){
    for(long i=0;i<h;i++) for(long j=0;j<h;j++)
        d[i*ldd+j]= sub? a[i*lda+j]-b[i*ldb+j] : a[i*lda+j]+b[i*ldb+j];
}

static void strassen(float *C,long ldc,const float *A,long lda,const float *B,long ldb,long n,long md){
    if(n<=md || (n&1)){ triple(C,ldc,A,lda,B,ldb,n); return; }
    long h=n/2;
    const float *A11=A,*A12=A+h,*A21=A+h*lda,*A22=A+h*lda+h;
    const float *B11=B,*B12=B+h,*B21=B+h*ldb,*B22=B+h*ldb+h;
    float *C11=C,*C12=C+h,*C21=C+h*ldc,*C22=C+h*ldc+h;
    float *M=malloc(sizeof(float)*h*h*7), *Ta=malloc(sizeof(float)*h*h), *Tb=malloc(sizeof(float)*h*h);
    float *M1=M,*M2=M+h*h,*M3=M+2*h*h,*M4=M+3*h*h,*M5=M+4*h*h,*M6=M+5*h*h,*M7=M+6*h*h;
    madd(Ta,h,A11,lda,A22,lda,h,0); madd(Tb,h,B11,ldb,B22,ldb,h,0); strassen(M1,h,Ta,h,Tb,h,h,md);
    madd(Ta,h,A21,lda,A22,lda,h,0);                                  strassen(M2,h,Ta,h,B11,ldb,h,md);
    madd(Tb,h,B12,ldb,B22,ldb,h,1);                                  strassen(M3,h,A11,lda,Tb,h,h,md);
    madd(Tb,h,B21,ldb,B11,ldb,h,1);                                  strassen(M4,h,A22,lda,Tb,h,h,md);
    madd(Ta,h,A11,lda,A12,lda,h,0);                                  strassen(M5,h,Ta,h,B22,ldb,h,md);
    madd(Ta,h,A21,lda,A11,lda,h,1); madd(Tb,h,B11,ldb,B12,ldb,h,0); strassen(M6,h,Ta,h,Tb,h,h,md);
    madd(Ta,h,A12,lda,A22,lda,h,1); madd(Tb,h,B21,ldb,B22,ldb,h,0); strassen(M7,h,Ta,h,Tb,h,h,md);
    for(long i=0;i<h;i++) for(long j=0;j<h;j++){
        float m1=M1[i*h+j],m2=M2[i*h+j],m3=M3[i*h+j],m4=M4[i*h+j],m5=M5[i*h+j],m6=M6[i*h+j],m7=M7[i*h+j];
        C11[i*ldc+j]=m1+m4-m5+m7;
        C12[i*ldc+j]=m3+m5;
        C21[i*ldc+j]=m2+m4;
        C22[i*ldc+j]=m1-m2+m3+m6;
    }
    free(M); free(Ta); free(Tb);
}

static void winograd(float *C,long ldc,const float *A,long lda,const float *B,long ldb,long n,long md){
    if(n<=md || (n&1)){ triple(C,ldc,A,lda,B,ldb,n); return; }
    long h=n/2;
    const float *A11=A,*A12=A+h,*A21=A+h*lda,*A22=A+h*lda+h;
    const float *B11=B,*B12=B+h,*B21=B+h*ldb,*B22=B+h*ldb+h;
    float *C11=C,*C12=C+h,*C21=C+h*ldc,*C22=C+h*ldc+h;
    float *M=malloc(sizeof(float)*h*h*7), *S=malloc(sizeof(float)*h*h), *T=malloc(sizeof(float)*h*h);
    float *M1=M,*M2=M+h*h,*M3=M+2*h*h,*M4=M+3*h*h,*M5=M+4*h*h,*M6=M+5*h*h,*M7=M+6*h*h;
    float *S1=malloc(sizeof(float)*h*h),*S2=malloc(sizeof(float)*h*h),*S3=malloc(sizeof(float)*h*h),*S4=malloc(sizeof(float)*h*h);
    /* S1=A21+A22; S2=S1-A11; S3=A11-A21; S4=A12-S2 */
    madd(S1,h,A21,lda,A22,lda,h,0);
    madd(S2,h,S1,h,A11,lda,h,1);
    madd(S3,h,A11,lda,A21,lda,h,1);
    madd(S4,h,A12,lda,S2,h,h,1);
    /* T1=B12-B11; T2=B22-T1; T3=B22-B12; T4=T2-B21  (reuse S,T bufs) */
    float *T1=malloc(sizeof(float)*h*h),*T2=malloc(sizeof(float)*h*h),*T3=malloc(sizeof(float)*h*h),*T4=malloc(sizeof(float)*h*h);
    madd(T1,h,B12,ldb,B11,ldb,h,1);
    madd(T2,h,B22,ldb,T1,h,h,1);
    madd(T3,h,B22,ldb,B12,ldb,h,1);
    madd(T4,h,T2,h,B21,ldb,h,1);
    winograd(M1,h,A11,lda,B11,ldb,h,md);
    winograd(M2,h,A12,lda,B21,ldb,h,md);
    winograd(M3,h,S4,h,B22,ldb,h,md);
    winograd(M4,h,A22,lda,T4,h,h,md);
    winograd(M5,h,S1,h,T1,h,h,md);
    winograd(M6,h,S2,h,T2,h,h,md);
    winograd(M7,h,S3,h,T3,h,h,md);
    for(long i=0;i<h;i++) for(long j=0;j<h;j++){
        float m1=M1[i*h+j],m2=M2[i*h+j],m3=M3[i*h+j],m4=M4[i*h+j],m5=M5[i*h+j],m6=M6[i*h+j],m7=M7[i*h+j];
        float u2=m1+m6, u3=u2+m7, u4=u2+m5;
        C11[i*ldc+j]=m1+m2;
        C12[i*ldc+j]=u4+m3;
        C21[i*ldc+j]=u3-m4;
        C22[i*ldc+j]=u3+m5;
    }
    free(M);free(S);free(T);free(S1);free(S2);free(S3);free(S4);free(T1);free(T2);free(T3);free(T4);
}

int main(int argc, char**argv){
    long dim = (argc>=2)? atol(argv[1]) : 256;
    const char *mode = (argc>=3)? argv[2] : "time";
    long md = MIN_DIM, n=dim, N=n*n;
    float *A=malloc(sizeof(float)*N), *B=malloc(sizeof(float)*N), *Cm=malloc(sizeof(float)*N);
    for(long i=0;i<n;i++) for(long j=0;j<n;j++){
        A[i*n+j]=(float)(((i*131+j*17)%97)*0.01 + 0.1);
        B[i*n+j]=(float)(((i*13+j*131)%89)*0.01 + 0.1);
    }

    if(strcmp(mode,"time")==0){
        double t;
        TIME_OP(t, triple(Cm,n,A,n,B,n,n));        printf("TIME,%s,%s,triple,%ld,%.9e\n",   PREC_NAME,BACKEND_NAME,dim,t);
        TIME_OP(t, block(Cm,n,A,n,B,n,n,md));      printf("TIME,%s,%s,block,%ld,%.9e\n",    PREC_NAME,BACKEND_NAME,dim,t);
        TIME_OP(t, strassen(Cm,n,A,n,B,n,n,md));   printf("TIME,%s,%s,strassen,%ld,%.9e\n", PREC_NAME,BACKEND_NAME,dim,t);
        TIME_OP(t, winograd(Cm,n,A,n,B,n,n,md));   printf("TIME,%s,%s,winograd,%ld,%.9e\n", PREC_NAME,BACKEND_NAME,dim,t);
    } else {
        mpfr_set_default_prec(REF_BITS);
        mpfr_t *Am=malloc(N*sizeof(mpfr_t)), *Bm=malloc(N*sizeof(mpfr_t)), *Ct=malloc(N*sizeof(mpfr_t));
        mpfr_t acc,prod,num,den,rel,maxrel,gen;
        mpfr_inits2(REF_BITS,acc,prod,num,den,rel,maxrel,gen,(mpfr_ptr)0);
        for(long i=0;i<n;i++) for(long j=0;j<n;j++){
            mpfr_set_d(gen,2.0+((i*131+j*17)%97)*0.01,MPFR_RNDN); mpfr_sqrt(gen,gen,MPFR_RNDN); A[i*n+j]=(float)mpfr_get_d(gen,MPFR_RNDN);
            mpfr_set_d(gen,3.0+((i*13+j*131)%89)*0.01,MPFR_RNDN); mpfr_sqrt(gen,gen,MPFR_RNDN); B[i*n+j]=(float)mpfr_get_d(gen,MPFR_RNDN);
        }
        for(long i=0;i<N;i++){ mpfr_init2(Am[i],REF_BITS); mpfr_set_d(Am[i],(double)A[i],MPFR_RNDN);
                               mpfr_init2(Bm[i],REF_BITS); mpfr_set_d(Bm[i],(double)B[i],MPFR_RNDN);
                               mpfr_init2(Ct[i],REF_BITS); }
        for(long i=0;i<n;i++) for(long j=0;j<n;j++){
            mpfr_set_zero(acc,1);
            for(long k=0;k<n;k++){ mpfr_mul(prod,Am[i*n+k],Bm[k*n+j],MPFR_RNDN); mpfr_add(acc,acc,prod,MPFR_RNDN); }
            mpfr_set(Ct[i*n+j],acc,MPFR_RNDN);
        }
        const char *names[4]={"triple","block","strassen","winograd"};
        for(int a=0;a<4;a++){
            switch(a){ case 0: triple(Cm,n,A,n,B,n,n); break; case 1: block(Cm,n,A,n,B,n,n,md); break;
                       case 2: strassen(Cm,n,A,n,B,n,n,md); break; case 3: winograd(Cm,n,A,n,B,n,n,md); break; }
            mpfr_set_zero(maxrel,1);
            for(long i=0;i<N;i++){
                mpfr_set_d(acc,(double)Cm[i],MPFR_RNDN);
                mpfr_sub(num,acc,Ct[i],MPFR_RNDN); mpfr_abs(num,num,MPFR_RNDN);
                mpfr_abs(den,Ct[i],MPFR_RNDN);
                if(mpfr_zero_p(den)) continue;
                mpfr_div(rel,num,den,MPFR_RNDN);
                if(mpfr_cmp(rel,maxrel)>0) mpfr_set(maxrel,rel,MPFR_RNDN);
            }
            printf("ACC,%s,%s,%s,%ld,%.6e\n", PREC_NAME,BACKEND_NAME,names[a],dim,mpfr_get_d(maxrel,MPFR_RNDN));
        }
        for(long i=0;i<N;i++){ mpfr_clear(Am[i]); mpfr_clear(Bm[i]); mpfr_clear(Ct[i]); }
        free(Am);free(Bm);free(Ct);
    }
    free(A);free(B);free(Cm);
    return 0;
}

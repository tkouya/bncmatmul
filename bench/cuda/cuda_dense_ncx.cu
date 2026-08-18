/*****************************************************************
 * cuda_dense_ncx.cu -- GPU vs CPU benchmark, NATIVE complex (cd / cf).
 *
 *   host I/O type is complex-double (CDMatrix/CDVector) for BOTH; the gcf device
 *   is single precision (downcast on transfer, verified to single tol).
 *
 *   cd matmul : GPU mul_gcdmatrix_dev      vs CPU serial mul_cdmatrix       (no OMP matmul)
 *   cd matvec : GPU mul_gcdmatrix_gcdvec   vs CPU OMP _bncomp_mul_cdmatrix_cdvec
 *   cf matmul : GPU mul_gcfmatrix_dev      vs CPU serial mul_cdmatrix       (double ref)
 *   cf matvec : GPU mul_gcfmatrix_gcfvec   vs CPU serial mul_cdmatrix_cdvec (double ref)
 *
 * CSV: RESULT,op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr
 *****************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

#include "cdlinear.h"
#include "gcdlinear.h"
#include "gcflinear.h"

#include <omp.h>
extern "C" void _bncomp_mul_cdmatrix_cdvec(CDVector, CDMatrix, CDVector);

static FILE *g_csv=nullptr;
static void emit_csv(const char*op,const char*prec,long d,long nz,double tc,double tg,const char*k,double re){
    if(!g_csv)return; fprintf(g_csv,"RESULT,%s,%s,%ld,%ld,%.9g,%.9g,%s,%.6e\n",op,prec,d,nz,tc,tg,k,re); fflush(g_csv);
}
using clk=std::chrono::high_resolution_clock;
static double secs(clk::time_point t0){return std::chrono::duration<double>(clk::now()-t0).count();}
template<class F> static double tg_(int r,F f){double b=1e30;for(int i=0;i<r;i++){auto t=clk::now();f();cudaDeviceSynchronize();double s=secs(t);if(s<b)b=s;}return b;}
template<class F> static double tc_(int r,F f){double b=1e30;for(int i=0;i<r;i++){auto t=clk::now();f();double s=secs(t);if(s<b)b=s;}return b;}

struct Opts{int reps=3,blocks=128,threads=128;long max_cpu=-1;std::vector<long> sizes;};
static void parse_sizes(const char*s,std::vector<long>&o){o.clear();const char*p=s;while(*p){char*e=0;long v=strtol(p,&e,10);if(e==p||v<=0)break;o.push_back(v);p=e;while(*p==','||*p==' ')++p;}}

/* build a double _Complex via GCC extension (C99 complex.h macros are unavailable
 * in nvcc C++ host mode) */
static inline double _Complex mkc(double re,double im){ double _Complex z; __real__ z=re; __imag__ z=im; return z; }

static void fill(CDMatrix A,CDMatrix B,CDVector x,long N){
    srand(20260513u);
    for(long i=0;i<N;i++){ set_cdvector_i(x,i,mkc((double)rand()/RAND_MAX*2-1,0));
        for(long j=0;j<N;j++){ set_cdmatrix_ij(A,i,j,mkc((double)rand()/RAND_MAX*2-1,0)); set_cdmatrix_ij(B,i,j,mkc((double)rand()/RAND_MAX*2-1,0));} }
}

static void bench_cd(const Opts&o){
    for(long N:o.sizes){
        bool rc=o.max_cpu<0||N<=o.max_cpu;
        CDMatrix A=init_cdmatrix(N,N),B=init_cdmatrix(N,N),Cs=init_cdmatrix(N,N),Cp=init_cdmatrix(N,N);
        CDVector x=init_cdvector(N),ys=init_cdvector(N),yp=init_cdvector(N);
        fill(A,B,x,N);
        GCDMatrix Ad=init_gcdmatrix_dev(N,N),Bd=init_gcdmatrix_dev(N,N),Cd=init_gcdmatrix_dev(N,N);
        GCDVector xd=init_gcdvector_dev(N),yd=init_gcdvector_dev(N);
        subst_gcdmatrix_dev_cdmat(Ad,A);subst_gcdmatrix_dev_cdmat(Bd,B);subst_gcdvector_dev_cdvec(xd,x);
        /* matmul: CPU serial */
        double tc=0,tg=0,re=0;
        mul_gcdmatrix_dev(Cd,Ad,Bd,o.blocks,o.threads);cudaDeviceSynchronize();
        tg=tg_(o.reps,[&]{mul_gcdmatrix_dev(Cd,Ad,Bd,o.blocks,o.threads);});
        if(rc){ mul_cdmatrix(Cs,A,B); tc=tc_(o.reps,[&]{mul_cdmatrix(Cs,A,B);});
            subst_cdmatrix_gcdmat_dev(Cp,Cd); double ns=normf_cdmatrix(Cs); sub_cdmatrix(Cp,Cs,Cp); double nd=normf_cdmatrix(Cp); re=ns!=0?nd/ns:nd; }
        emit_csv("matmul","cd",N,0,tc,tg,rc?"serial":"none",rc?re:-1);
        /* matvec: CPU OMP */
        tc=tg=re=0;
        mul_gcdmatrix_gcdvec(yd,Ad,xd,o.blocks,o.threads);cudaDeviceSynchronize();
        tg=tg_(o.reps,[&]{mul_gcdmatrix_gcdvec(yd,Ad,xd,o.blocks,o.threads);});
        _bncomp_mul_cdmatrix_cdvec(ys,A,x); tc=tc_(o.reps,[&]{_bncomp_mul_cdmatrix_cdvec(ys,A,x);});
        subst_cdvector_gcdvec_dev(yp,yd); { double ns=norm2_cdvector(ys); sub_cdvector(yp,ys,yp); double nd=norm2_cdvector(yp); re=ns!=0?nd/ns:nd; }
        emit_csv("matvec","cd",N,0,tc,tg,"omp",re);
        free_gcdmatrix_dev(Ad);free_gcdmatrix_dev(Bd);free_gcdmatrix_dev(Cd);free_gcdvector_dev(xd);free_gcdvector_dev(yd);
        free_cdmatrix(A);free_cdmatrix(B);free_cdmatrix(Cs);free_cdmatrix(Cp);free_cdvector(x);free_cdvector(ys);free_cdvector(yp);
    }
}

static void bench_cf(const Opts&o){
    for(long N:o.sizes){
        bool rc=o.max_cpu<0||N<=o.max_cpu;
        CDMatrix A=init_cdmatrix(N,N),B=init_cdmatrix(N,N),Cs=init_cdmatrix(N,N),Cp=init_cdmatrix(N,N);
        CDVector x=init_cdvector(N),ys=init_cdvector(N),yp=init_cdvector(N);
        fill(A,B,x,N);
        GCFMatrix Ad=init_gcfmatrix_dev(N,N),Bd=init_gcfmatrix_dev(N,N),Cd=init_gcfmatrix_dev(N,N);
        GCFVector xd=init_gcfvector_dev(N),yd=init_gcfvector_dev(N);
        subst_gcfmatrix_dev_cdmat(Ad,A);subst_gcfmatrix_dev_cdmat(Bd,B);subst_gcfvector_dev_cdvec(xd,x);
        double tc=0,tg=0,re=0;
        mul_gcfmatrix_dev(Cd,Ad,Bd,o.blocks,o.threads);cudaDeviceSynchronize();
        tg=tg_(o.reps,[&]{mul_gcfmatrix_dev(Cd,Ad,Bd,o.blocks,o.threads);});
        if(rc){ mul_cdmatrix(Cs,A,B); tc=tc_(o.reps,[&]{mul_cdmatrix(Cs,A,B);});
            subst_cdmatrix_gcfmat_dev(Cp,Cd); double ns=normf_cdmatrix(Cs); sub_cdmatrix(Cp,Cs,Cp); double nd=normf_cdmatrix(Cp); re=ns!=0?nd/ns:nd; }
        emit_csv("matmul","cf",N,0,tc,tg,rc?"serial":"none",rc?re:-1);
        tc=tg=re=0;
        mul_gcfmatrix_gcfvec(yd,Ad,xd,o.blocks,o.threads);cudaDeviceSynchronize();
        tg=tg_(o.reps,[&]{mul_gcfmatrix_gcfvec(yd,Ad,xd,o.blocks,o.threads);});
        mul_cdmatrix_cdvec(ys,A,x); tc=tc_(o.reps,[&]{mul_cdmatrix_cdvec(ys,A,x);});
        subst_cdvector_gcfvec_dev(yp,yd); { double ns=norm2_cdvector(ys); sub_cdvector(yp,ys,yp); double nd=norm2_cdvector(yp); re=ns!=0?nd/ns:nd; }
        emit_csv("matvec","cf",N,0,tc,tg,"serial",re);
        free_gcfmatrix_dev(Ad);free_gcfmatrix_dev(Bd);free_gcfmatrix_dev(Cd);free_gcfvector_dev(xd);free_gcfvector_dev(yd);
        free_cdmatrix(A);free_cdmatrix(B);free_cdmatrix(Cs);free_cdmatrix(Cp);free_cdvector(x);free_cdvector(ys);free_cdvector(yp);
    }
}

int main(int argc,char**argv){
    Opts o; o.sizes={128,256,512,1024,2048};
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc)o.reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--blocks")&&i+1<argc)o.blocks=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)o.threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-cpu")&&i+1<argc)o.max_cpu=atol(argv[++i]);
        else if(!strcmp(argv[i],"--sizes")&&i+1<argc)parse_sizes(argv[++i],o.sizes);
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    const char*env=getenv("OMP_NUM_THREADS");int nt=env?atoi(env):omp_get_max_threads();if(nt<1)nt=omp_get_max_threads();omp_set_num_threads(nt);
    g_csv=stdout; cudaSetDevice(0); cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf(" native complex : device %s sm_%d%d  OMP=%d (cd matvec); rest serial\n",p.name,p.major,p.minor,nt);
    bench_cd(o); bench_cf(o);
    return 0;
}

/*****************************************************************
 * cuda_dense_nr.cu -- GPU vs CPU(OpenMP) benchmark, NATIVE real (double / float).
 *
 *   matmul : C = A*B      GPU mul_g{d,f}matrix_dev    vs CPU _bncomp_mul_dmatrix / mul_fmatrix
 *   matvec : y = A*x      GPU mul_g{d,f}matrix_g{d,f}vec vs CPU _bncomp_mul_dmatrix_dvec / mul_fmatrix_dvec
 *   axpy   : c = a*x      GPU cmul_g{d,f}vector_dev   vs CPU _bncomp_cmul_dvector / cmul_fvector
 *
 *   double : CPU baseline = OpenMP (_bncomp_*).            cpu_kind = "omp"
 *   float  : no OpenMP routine in the library -> CPU = serial. cpu_kind = "serial"
 *
 * CSV (stdout):  RESULT,op,prec,dim,nnz,cpu_time,gpu_time,cpu_kind,relerr
 *****************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include <cuda_runtime.h>

#include "dlinear.h"
#include "flinear.h"
#include "gdlinear.h"
#include "gflinear.h"

#include <omp.h>

/* OpenMP native-double entry points (C linkage; float has none) */
extern "C" {
void _bncomp_mul_dmatrix(DMatrix, DMatrix, DMatrix);
void _bncomp_mul_dmatrix_dvec(DVector, DMatrix, DVector);
void _bncomp_cmul_dvector(DVector, double, DVector);
}
/* the library exports float matvec under the _dvec name (see flinear note) */
extern "C" void mul_fmatrix_dvec(FVector, FMatrix, FVector);

static FILE *g_csv = nullptr;
static void emit_csv(const char *op, const char *prec, long dim, long nnz,
                     double tcpu, double tgpu, const char *kind, double relerr){
    if(!g_csv) return;
    fprintf(g_csv,"RESULT,%s,%s,%ld,%ld,%.9g,%.9g,%s,%.6e\n",op,prec,dim,nnz,tcpu,tgpu,kind,relerr);
    fflush(g_csv);
}

using clk = std::chrono::high_resolution_clock;
static double secs(clk::time_point t0){ return std::chrono::duration<double>(clk::now()-t0).count(); }
template<class F> static double tmin_gpu(int r, F f){ double b=1e30; for(int i=0;i<r;i++){auto t=clk::now(); f(); cudaDeviceSynchronize(); double s=secs(t); if(s<b)b=s;} return b; }
template<class F> static double tmin_cpu(int r, F f){ double b=1e30; for(int i=0;i<r;i++){auto t=clk::now(); f(); double s=secs(t); if(s<b)b=s;} return b; }

struct Opts { int reps=3, blocks=128, threads=128; long max_cpu=-1; std::vector<long> sizes, axpy_sizes; };

/*------------------------- double (OpenMP CPU) -------------------------*/
static void bench_d(const Opts &o)
{
    for (long N : o.sizes) {
        bool run_cpu = o.max_cpu < 0 || N <= o.max_cpu;
        DMatrix A=init_dmatrix(N,N), B=init_dmatrix(N,N), Cs=init_dmatrix(N,N), Cp=init_dmatrix(N,N);
        DVector x=init_dvector(N), ys=init_dvector(N), yp=init_dvector(N);
        srand(20260513u);
        for(long i=0;i<N;i++){ set_dvector_i(x,i,(double)rand()/RAND_MAX*2-1);
            for(long j=0;j<N;j++){ set_dmatrix_ij(A,i,j,(double)rand()/RAND_MAX*2-1); set_dmatrix_ij(B,i,j,(double)rand()/RAND_MAX*2-1);} }

        GDMatrix Ad=init_gdmatrix_dev(N,N), Bd=init_gdmatrix_dev(N,N), Cd=init_gdmatrix_dev(N,N);
        GDVector xd=init_gdvector_dev(N), yd=init_gdvector_dev(N);
        subst_gdmatrix_dev_dmat(Ad,A); subst_gdmatrix_dev_dmat(Bd,B); subst_gdvector_dev_dvec(xd,x);

        /* matmul */
        double tc=0,tg=0,re=0;
        mul_gdmatrix_dev(Cd,Ad,Bd,o.blocks,o.threads); cudaDeviceSynchronize();
        tg=tmin_gpu(o.reps,[&]{ mul_gdmatrix_dev(Cd,Ad,Bd,o.blocks,o.threads); });
        if(run_cpu){ _bncomp_mul_dmatrix(Cs,A,B); tc=tmin_cpu(o.reps,[&]{ _bncomp_mul_dmatrix(Cs,A,B); });
            subst_dmatrix_gdmat_dev(Cp,Cd); double ns=normf_dmatrix(Cs); sub_dmatrix(Cp,Cs,Cp); double nd=normf_dmatrix(Cp); re=ns!=0?nd/ns:nd; }
        emit_csv("matmul","d",N,0,tc,tg,run_cpu?"omp":"none",run_cpu?re:-1);

        /* matvec */
        tc=tg=re=0;
        mul_gdmatrix_gdvec(yd,Ad,xd,o.blocks,o.threads); cudaDeviceSynchronize();
        tg=tmin_gpu(o.reps,[&]{ mul_gdmatrix_gdvec(yd,Ad,xd,o.blocks,o.threads); });
        _bncomp_mul_dmatrix_dvec(ys,A,x); tc=tmin_cpu(o.reps,[&]{ _bncomp_mul_dmatrix_dvec(ys,A,x); });
        subst_dvector_gdvec_dev(yp,yd); { double ns=norm2_dvector(ys); sub_dvector(yp,ys,yp); double nd=norm2_dvector(yp); re=ns!=0?nd/ns:nd; }
        emit_csv("matvec","d",N,0,tc,tg,"omp",re);

        free_gdmatrix_dev(Ad); free_gdmatrix_dev(Bd); free_gdmatrix_dev(Cd); free_gdvector_dev(xd); free_gdvector_dev(yd);
        free_dmatrix(A); free_dmatrix(B); free_dmatrix(Cs); free_dmatrix(Cp); free_dvector(x); free_dvector(ys); free_dvector(yp);
    }
    /* axpy on long vectors */
    for (long L : o.axpy_sizes) {
        DVector a=init_dvector(L), cs=init_dvector(L), cp=init_dvector(L);
        srand(7u); for(long i=0;i<L;i++) set_dvector_i(a,i,(double)rand()/RAND_MAX*2-1);
        double val=0.7283105;
        GDVector ad=init_gdvector_dev(L), cd=init_gdvector_dev(L); subst_gdvector_dev_dvec(ad,a);
        cmul_gdvector_dev(cd,val,ad,o.blocks,o.threads); cudaDeviceSynchronize();
        double tg=tmin_gpu(o.reps,[&]{ cmul_gdvector_dev(cd,val,ad,o.blocks,o.threads); });
        _bncomp_cmul_dvector(cs,val,a); double tc=tmin_cpu(o.reps,[&]{ _bncomp_cmul_dvector(cs,val,a); });
        subst_dvector_gdvec_dev(cp,cd); double ns=norm2_dvector(cs); sub_dvector(cp,cs,cp); double nd=norm2_dvector(cp); double re=ns!=0?nd/ns:nd;
        emit_csv("axpy","d",L,0,tc,tg,"omp",re);
        free_gdvector_dev(ad); free_gdvector_dev(cd); free_dvector(a); free_dvector(cs); free_dvector(cp);
    }
}

/*------------------------- float (serial CPU baseline) -------------------------*/
static void bench_f(const Opts &o)
{
    for (long N : o.sizes) {
        bool run_cpu = o.max_cpu < 0 || N <= o.max_cpu;
        FMatrix A=init_fmatrix(N,N), B=init_fmatrix(N,N), Cs=init_fmatrix(N,N), Cp=init_fmatrix(N,N);
        FVector x=init_fvector(N), ys=init_fvector(N), yp=init_fvector(N);
        srand(20260513u);
        for(long i=0;i<N;i++){ set_fvector_i(x,i,(float)((double)rand()/RAND_MAX*2-1));
            for(long j=0;j<N;j++){ set_fmatrix_ij(A,i,j,(float)((double)rand()/RAND_MAX*2-1)); set_fmatrix_ij(B,i,j,(float)((double)rand()/RAND_MAX*2-1)); } }

        GFMatrix Ad=init_gfmatrix_dev(N,N), Bd=init_gfmatrix_dev(N,N), Cd=init_gfmatrix_dev(N,N);
        GFVector xd=init_gfvector_dev(N), yd=init_gfvector_dev(N);
        subst_gfmatrix_dev_fmat(Ad,A); subst_gfmatrix_dev_fmat(Bd,B); subst_gfvector_dev_fvec(xd,x);

        double tc=0,tg=0,re=0;
        mul_gfmatrix_dev(Cd,Ad,Bd,o.blocks,o.threads); cudaDeviceSynchronize();
        tg=tmin_gpu(o.reps,[&]{ mul_gfmatrix_dev(Cd,Ad,Bd,o.blocks,o.threads); });
        if(run_cpu){ mul_fmatrix(Cs,A,B); tc=tmin_cpu(o.reps,[&]{ mul_fmatrix(Cs,A,B); });
            subst_fmatrix_gfmat_dev(Cp,Cd); float ns=normf_fmatrix(Cs); sub_fmatrix(Cp,Cs,Cp); float nd=normf_fmatrix(Cp); re=ns!=0?(double)nd/ns:nd; }
        emit_csv("matmul","f",N,0,tc,tg,run_cpu?"serial":"none",run_cpu?re:-1);

        tc=tg=re=0;
        mul_gfmatrix_gfvec(yd,Ad,xd,o.blocks,o.threads); cudaDeviceSynchronize();
        tg=tmin_gpu(o.reps,[&]{ mul_gfmatrix_gfvec(yd,Ad,xd,o.blocks,o.threads); });
        mul_fmatrix_dvec(ys,A,x); tc=tmin_cpu(o.reps,[&]{ mul_fmatrix_dvec(ys,A,x); });
        subst_fvector_gfvec_dev(yp,yd); { float ns=norm2_fvector(ys); sub_fvector(yp,ys,yp); float nd=norm2_fvector(yp); re=ns!=0?(double)nd/ns:nd; }
        emit_csv("matvec","f",N,0,tc,tg,"serial",re);

        free_gfmatrix_dev(Ad); free_gfmatrix_dev(Bd); free_gfmatrix_dev(Cd); free_gfvector_dev(xd); free_gfvector_dev(yd);
        free_fmatrix(A); free_fmatrix(B); free_fmatrix(Cs); free_fmatrix(Cp); free_fvector(x); free_fvector(ys); free_fvector(yp);
    }
    for (long L : o.axpy_sizes) {
        FVector a=init_fvector(L), cs=init_fvector(L), cp=init_fvector(L);
        srand(7u); for(long i=0;i<L;i++) set_fvector_i(a,i,(float)((double)rand()/RAND_MAX*2-1));
        float val=0.7283105f;
        GFVector ad=init_gfvector_dev(L), cd=init_gfvector_dev(L); subst_gfvector_dev_fvec(ad,a);
        cmul_gfvector_dev(cd,val,ad,o.blocks,o.threads); cudaDeviceSynchronize();
        double tg=tmin_gpu(o.reps,[&]{ cmul_gfvector_dev(cd,val,ad,o.blocks,o.threads); });
        cmul_fvector(cs,val,a); double tc=tmin_cpu(o.reps,[&]{ cmul_fvector(cs,val,a); });
        subst_fvector_gfvec_dev(cp,cd); float ns=norm2_fvector(cs); sub_fvector(cp,cs,cp); float nd=norm2_fvector(cp); double re=ns!=0?(double)nd/ns:nd;
        emit_csv("axpy","f",L,0,tc,tg,"serial",re);
        free_gfvector_dev(ad); free_gfvector_dev(cd); free_fvector(a); free_fvector(cs); free_fvector(cp);
    }
}

static void parse_sizes(const char *s, std::vector<long> &out){ out.clear(); const char*p=s; while(*p){char*e=0; long v=strtol(p,&e,10); if(e==p||v<=0)break; out.push_back(v); p=e; while(*p==','||*p==' ')++p;} }

int main(int argc, char **argv)
{
    Opts o; o.sizes={128,256,512,1024,2048}; o.axpy_sizes={16384,65536,262144,1048576};
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc) o.reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--blocks")&&i+1<argc) o.blocks=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")&&i+1<argc) o.threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-cpu")&&i+1<argc) o.max_cpu=atol(argv[++i]);
        else if(!strcmp(argv[i],"--sizes")&&i+1<argc) parse_sizes(argv[++i],o.sizes);
        else if(!strcmp(argv[i],"--axpy-sizes")&&i+1<argc) parse_sizes(argv[++i],o.axpy_sizes);
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    const char *env=getenv("OMP_NUM_THREADS"); int nt=env?atoi(env):omp_get_max_threads(); if(nt<1)nt=omp_get_max_threads(); omp_set_num_threads(nt);
    g_csv=stdout;
    cudaSetDevice(0); cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    printf(" device: %s (sm_%d%d)  CPU OpenMP threads=%d (double); float CPU=serial\n",p.name,p.major,p.minor,nt);
    bench_d(o);
    bench_f(o);
    return 0;
}

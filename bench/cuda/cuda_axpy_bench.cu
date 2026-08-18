/****************************************************************************/
/* cuda_axpy_bench.cu : GPU AXPY (c = a + alpha*b) timing for BNCmatmul      */
/*   precisions : double, float, dd, td, qd, ds, ts, qs                      */
/*   The library exposes no fused add_cmul on the GPU (only cmul = val*a and */
/*   add = a+b), so we provide our own FUSED axpy kernels using the same     */
/*   device EFT operators (operator+, operator*) the library kernels use.    */
/*   All kernels are grid-stride, so a 32-thread launch still covers all N.  */
/*                                                                            */
/*   Two launch configs are timed per size:                                  */
/*     - "32 total CUDA threads"  : 1 block x 32 threads  (== CPU max=32)     */
/*     - "GPU max threads"        : 256 threads/block, ceil(N/256) blocks     */
/*       (one thread per element -> full occupancy on the H100)              */
/*                                                                            */
/*   Output (CSV, matches the CPU harness schema):                           */
/*     backend,nthreads,family,type,bits,N,seconds,iters,mflops              */
/****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

#include <qd/qd_real.h>
#include <qd/fpu.h>
#include "rds.h"

#include <cuda_runtime.h>
#include <vector_types.h>

#include "gqd_type.h"
#include "gddlinear.h"
#include "gtdlinear.h"
#include "gdslinear.h"
#include "gtslinear.h"
#include "ddlinear.h"
#include "tdlinear.h"
#include "qdlinear.h"
#include "dslinear.h"
#include "tslinear.h"
#include "qslinear.h"

extern "C" {
void add_cmul_ddvector(DDVector, DDVector, double[], DDVector);
void add_cmul_tdvector(TDVector, TDVector, double[], TDVector);
void add_cmul_qdvector(QDVector, QDVector, double[], QDVector);
void add_cmul_dsvector(DSVector, DSVector, float[],  DSVector);
void add_cmul_tsvector(TSVector, TSVector, float[],  TSVector);
void add_cmul_qsvector(QSVector, QSVector, float[],  QSVector);
}

using clk = std::chrono::high_resolution_clock;
static double secs_since(clk::time_point t0){
    return std::chrono::duration<double>(clk::now() - t0).count();
}
static double g_cap = 2.0;   /* stop repeating once one call exceeds this */
template<typename Fn>
static double measure_min_gpu(int reps, Fn fn){
    double best = 1e30;
    for (int i = 0; i < reps; ++i){
        auto t0 = clk::now();
        fn(); cudaDeviceSynchronize();
        double s = secs_since(t0);
        if (s < best) best = s;
        if (s >= g_cap) break;   /* slow kernel: one timed call is enough */
    }
    return best;
}

static int   g_reps = 7;
static FILE *g_out  = nullptr;   /* CSV sink (defaults to stdout) */
static void emit(const char *fam,const char *type,long bits,long nthreads,
                 long N,double sec){
    double mflops = (sec>0)?(2.0*(double)N/sec/1.0e6):0.0;
    fprintf(g_out,"cuda,%ld,%s,%s,%ld,%ld,%.6e,%d,%.4f\n",
            nthreads, fam, type, bits, N, sec, g_reps, mflops);
    fflush(g_out);
}

/*=========================== fused AXPY kernels ==========================*/
#define AXPY_KERNEL(NAME, RT)                                                 \
__global__ void NAME(RT *c, const RT *a, RT val, const RT *b, long n){        \
    long i = threadIdx.x + (long)blockIdx.x*blockDim.x;                       \
    long stride = (long)blockDim.x*gridDim.x;                                 \
    for(; i<n; i+=stride) c[i] = a[i] + val*b[i];                             \
}
AXPY_KERNEL(axpy_k_gdd, gdd_real)
AXPY_KERNEL(axpy_k_gtd, gtd_real)
AXPY_KERNEL(axpy_k_gqd, gqd_real)
AXPY_KERNEL(axpy_k_gds, gds_real)
AXPY_KERNEL(axpy_k_gts, gts_real)
AXPY_KERNEL(axpy_k_gqs, gqs_real)
/* native double/float */
__global__ void axpy_k_d(double *c,const double *a,double val,const double *b,long n){
    long i=threadIdx.x+(long)blockIdx.x*blockDim.x, st=(long)blockDim.x*gridDim.x;
    for(;i<n;i+=st) c[i]=a[i]+val*b[i];
}
__global__ void axpy_k_f(float *c,const float *a,float val,const float *b,long n){
    long i=threadIdx.x+(long)blockIdx.x*blockDim.x, st=(long)blockDim.x*gridDim.x;
    for(;i<n;i+=st) c[i]=a[i]+val*b[i];
}

/*=========================== EFT precision bench =========================*/
/* P,GP,V,GV,GR,S,SZ,VSFX,KERN,BITS,FAM */
static long g_verify_max = 65536;   /* run the (slow per-element) relerr check only up to here */
#define DEFINE_EFT_BENCH(P,V,GV,GR,S,SZ,VSFX,KERN,BITS,FAM)                   \
static void bench_##P(long N, long nthreads_total, long blocks, int threads,  \
                      double *relerr)                                         \
{                                                                            \
    V a_cpu=init_##P##vector(N), b_cpu=init_##P##vector(N);                   \
    S val[SZ]; for(int k=0;k<SZ;k++) val[k]=(S)0; val[0]=(S)1.5;              \
    for(long i=0;i<N;i++){                                                    \
        set_##P##vector_i_##VSFX(a_cpu,i,(double)((i%9+1)*0.5));              \
        set_##P##vector_i_##VSFX(b_cpu,i,(double)((i%4+1)*0.5));              \
    }                                                                        \
    /* pack host SoA -> AoS gprec_real buffer, then ONE bulk H2D copy each */ \
    GR *ha=(GR*)malloc(N*sizeof(GR)), *hb=(GR*)malloc(N*sizeof(GR));          \
    for(long i=0;i<N;i++){                                                    \
        S *la=reinterpret_cast<S*>(&ha[i]), *lb=reinterpret_cast<S*>(&hb[i]); \
        for(int k=0;k<SZ;k++){ la[k]=a_cpu->element[k][i]; lb[k]=b_cpu->element[k][i]; } \
    }                                                                        \
    GV a_dev=init_g##P##vector_dev(N), b_dev=init_g##P##vector_dev(N),        \
       c_dev=init_g##P##vector_dev(N);                                       \
    cudaMemcpy(a_dev->element,ha,N*sizeof(GR),cudaMemcpyHostToDevice);        \
    cudaMemcpy(b_dev->element,hb,N*sizeof(GR),cudaMemcpyHostToDevice);        \
    GR gval; memset(&gval,0,sizeof(gval));                                    \
    { S *l=reinterpret_cast<S*>(&gval); for(int k=0;k<SZ;k++) l[k]=val[k]; }  \
    dim3 gb(blocks), gt(threads);                                             \
    KERN<<<gb,gt>>>(c_dev->element,a_dev->element,gval,b_dev->element,N);     \
    cudaDeviceSynchronize();                                                  \
    double t=measure_min_gpu(g_reps,[&]{                                      \
        KERN<<<gb,gt>>>(c_dev->element,a_dev->element,gval,b_dev->element,N); \
    });                                                                       \
    *relerr=-1.0;                                                             \
    if(N<=g_verify_max){ /* correctness vs CPU serial add_cmul (small N only)*/\
      V cs_cpu=init_##P##vector(N), cp_cpu=init_##P##vector(N);               \
      add_cmul_##P##vector(cs_cpu,a_cpu,val,b_cpu);                          \
      subst_##P##vector_g##P##vec_dev(cp_cpu,c_dev);                         \
      S ns[SZ],nd[SZ]; norm2_##P##vector(ns,cs_cpu);                          \
      sub_##P##vector(cp_cpu,cs_cpu,cp_cpu); norm2_##P##vector(nd,cp_cpu);    \
      double n_s=(double)ns[0], n_d=(double)nd[0];                            \
      *relerr=(n_s!=0.0)?(n_d/n_s):n_d;                                       \
      free_##P##vector(cs_cpu); free_##P##vector(cp_cpu); }                   \
    emit(FAM,#P,BITS,nthreads_total,N,t);                                     \
    free(ha); free(hb);                                                       \
    free_g##P##vector_dev(a_dev); free_g##P##vector_dev(b_dev);              \
    free_g##P##vector_dev(c_dev);                                             \
    free_##P##vector(a_cpu); free_##P##vector(b_cpu);                         \
}
DEFINE_EFT_BENCH(dd,DDVector,GDDVector,gdd_real,double,DDSIZE,d,axpy_k_gdd,106,"double-based")
DEFINE_EFT_BENCH(td,TDVector,GTDVector,gtd_real,double,TDSIZE,d,axpy_k_gtd,159,"double-based")
DEFINE_EFT_BENCH(qd,QDVector,GQDVector,gqd_real,double,QDSIZE,d,axpy_k_gqd,212,"double-based")
DEFINE_EFT_BENCH(ds,DSVector,GDSVector,gds_real,float, DSSIZE,d,axpy_k_gds, 48,"float-based")
DEFINE_EFT_BENCH(ts,TSVector,GTSVector,gts_real,float, TSSIZE,f,axpy_k_gts, 72,"float-based")
DEFINE_EFT_BENCH(qs,QSVector,GQSVector,gqs_real,float, QSSIZE,f,axpy_k_gqs, 96,"float-based")

/*=========================== native d/f bench ===========================*/
template<typename T, typename K>
static void bench_native(const char *name,const char *fam,long bits,
                         long N,long nthreads_total,long blocks,int threads,K kern){
    T *a=(T*)malloc(N*sizeof(T)),*b=(T*)malloc(N*sizeof(T));
    for(long i=0;i<N;i++){ a[i]=(T)((i%9+1)*0.5); b[i]=(T)((i%4+1)*0.5); }
    T *da,*db,*dc; cudaMalloc(&da,N*sizeof(T)); cudaMalloc(&db,N*sizeof(T)); cudaMalloc(&dc,N*sizeof(T));
    cudaMemcpy(da,a,N*sizeof(T),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,N*sizeof(T),cudaMemcpyHostToDevice);
    T val=(T)1.5; dim3 gb(blocks),gt(threads);
    kern<<<gb,gt>>>(dc,da,val,db,N); cudaDeviceSynchronize();
    double t=measure_min_gpu(g_reps,[&]{ kern<<<gb,gt>>>(dc,da,val,db,N); });
    emit(fam,name,bits,nthreads_total,N,t);
    cudaFree(da); cudaFree(db); cudaFree(dc); free(a); free(b);
}

int main(int argc,char**argv){
    std::vector<long> sizes = {4096,16384,65536,262144,1048576,4194304};
    long cap32 = 65536;    /* skip the (single-warp, pathologically slow) 32-thread
                              config above this N */
    int dev=0; const char *outpath=nullptr;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc) g_reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dev")&&i+1<argc) dev=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--cap32")&&i+1<argc) cap32=atol(argv[++i]);
        else if(!strcmp(argv[i],"--out")&&i+1<argc) outpath=argv[++i];
        else if(!strcmp(argv[i],"--sizes")&&i+1<argc){
            sizes.clear(); char*p=argv[++i]; while(*p){char*e; long v=strtol(p,&e,10);
                if(e==p)break; if(v>0)sizes.push_back(v); p=e; while(*p==','||*p==' ')++p;} }
    }
    g_out = stdout;
    if(outpath){ g_out = fopen(outpath,"w"); if(!g_out){perror(outpath); return 1;} }
    unsigned int cw; fpu_fix_start(&cw);
    cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
    fprintf(stderr,"# GPU AXPY: %s sm_%d%d  reps=%d\n",prop.name,prop.major,prop.minor,g_reps);
    GDDStart(dev); GTDStart(dev); GQDStart(dev);
    GDSStart(dev); GTSStart(dev); GQSStart(dev);

    fprintf(g_out,"backend,nthreads,family,type,bits,N,seconds,iters,mflops\n");
    fflush(g_out);
    const int TPB=256;             /* threads/block for the "max" config */
    for(long N : sizes){
        long mblocks = (N + TPB - 1)/TPB;     /* one thread per element  */
        long mthreads = mblocks*(long)TPB;    /* grid thread count       */
        double re;
        /* ---- config A: 32 total CUDA threads (1 block x 32) ---- */
        if(N <= cap32){
        bench_native<double>("double","double-based",53,N,32,1,32,axpy_k_d);
        bench_native<float> ("float", "float-based", 24,N,32,1,32,axpy_k_f);
        bench_dd(N,32,1,32,&re); bench_td(N,32,1,32,&re); bench_qd(N,32,1,32,&re);
        bench_ds(N,32,1,32,&re); bench_ts(N,32,1,32,&re); bench_qs(N,32,1,32,&re);
        }
        /* ---- config B: GPU max threads (one thread per element) ---- */
        bench_native<double>("double","double-based",53,N,mthreads,mblocks,TPB,axpy_k_d);
        bench_native<float> ("float", "float-based", 24,N,mthreads,mblocks,TPB,axpy_k_f);
        bench_dd(N,mthreads,mblocks,TPB,&re); bench_td(N,mthreads,mblocks,TPB,&re);
        bench_qd(N,mthreads,mblocks,TPB,&re);
        bench_ds(N,mthreads,mblocks,TPB,&re); bench_ts(N,mthreads,mblocks,TPB,&re);
        bench_qs(N,mthreads,mblocks,TPB,&re);
        fprintf(stderr,"  N=%ld done (last relerr=%.2e)\n",N,re);
    }
    GDDEnd(); GTDEnd(); GQDEnd(); GDSEnd(); GTSEnd(); GQSEnd();
    fpu_fix_end(&cw);
    return 0;
}

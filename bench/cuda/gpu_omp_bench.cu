/*****************************************************************************
 * gpu_omp_bench.cu  --  DENSE basic-linear GPU(gdtq) vs CPU-OMP(_bncomp_)
 * ---------------------------------------------------------------------------
 * matvec : y = A*x   GPU mul_g<P>matrix_g<P>vec   CPU _bncomp_mul_<P>matrix_<P>vec
 * matmul : C = A*B   GPU mul_g<P>matrix_dev       CPU _bncomp_mul_<P>matrix
 * over the six EFT layers dd/td/qd (double-based), ds/ts/qs (single-based).
 *
 * GPU time : kernel-only (device-resident, cudaDeviceSynchronize), min/reps.
 * CPU time : min/reps, OpenMP (OMP_NUM_THREADS or all cores).
 * relerr   : max rel. error GPU vs CPU on reconstructed leading value (N<=cap).
 *
 * CSV: op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops
 * (SpMV is a separate binary: gpu_spmv_bench.cu -- see build_gpu_omp.sh)
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>

#include <qd/qd_real.h>
#include <qd/fpu.h>
#include "rds.h"

#include <cuda_runtime.h>
#include <vector_types.h>
#include <omp.h>

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

#define USE_DDLINEAR
#define USE_TDLINEAR
#define USE_QDLINEAR
#define USE_DSLINEAR
#define USE_TSLINEAR
#define USE_QSLINEAR
#include "bncomp.h"     /* _bncomp_* OpenMP entry points */

using clk = std::chrono::high_resolution_clock;
static double secs_since(clk::time_point t0){ return std::chrono::duration<double>(clk::now()-t0).count(); }
static double g_cap = 8.0;
template<typename Fn> static double measure_min(int reps, Fn fn, bool gpu){
    double best=1e30;
    for(int i=0;i<reps;++i){ auto t0=clk::now(); fn(); if(gpu) cudaDeviceSynchronize();
        double s=secs_since(t0); if(s<best)best=s; if(s>=g_cap)break; }
    return best;
}
static FILE *g_out=nullptr; static int g_reps=3; static long g_verify_cap=1024;
static int g_blocks=256,g_threads=256; static long g_maxcpu_mm=2048;
static void emit(const char*op,const char*fam,const char*type,long bits,long N,long nnz,
                 double gpu,double cpu,const char*cpub,double relerr,double flop){
    double sp=(gpu>0)?cpu/gpu:0, gmf=(gpu>0)?flop/gpu/1e6:0, cmf=(cpu>0)?flop/cpu/1e6:0;
    fprintf(g_out,"%s,%s,%s,%ld,%ld,%ld,%.6e,%.6e,%s,%.3f,%.3e,%.4f,%.4f\n",
            op,fam,type,bits,N,nnz,gpu,cpu,cpub,sp,relerr,gmf,cmf);
    fflush(g_out);
}
static double frand(){ return (double)rand()/(double)RAND_MAX*2.0-1.0; }
/* Fast device alloc: library init_g*matrix_dev zeros element-by-element (N^2
 * cudaMemcpy -> ~10s). We overwrite via bulk H2D, so just malloc struct + one
 * cudaMalloc. Layout: matrix{long row_dim,col_dim; T* element}; vector{long dim; T* element}. */
#define FAST_MAT(GM,GR,var,N) GM var=(GM)malloc(sizeof(*var)); var->row_dim=(N); var->col_dim=(N); \
    cudaMalloc((void**)&var->element,(size_t)(N)*(N)*sizeof(GR))
#define FAST_VEC(GV,GR,var,N) GV var=(GV)malloc(sizeof(*var)); var->dim=(N); \
    cudaMalloc((void**)&var->element,(size_t)(N)*sizeof(GR))
#define FAST_FREE(var) do{ cudaFree(var->element); free(var); }while(0)

#define DEFINE_DENSE(P,V,M,GV,GM,GR,S,SZ,MSFX,VSFX,BITS,FAM,OMP_MV,OMP_MM)            \
static void mv_##P(long N){                                                          \
    bool _T=getenv("TSETUP")!=0; auto _tp=clk::now();                                 \
    M A=init_##P##matrix(N,N); V x=init_##P##vector(N); V yc=init_##P##vector(N);     \
    if(_T){fprintf(stderr,"  [%s N=%ld] alloc %.2fs\n",#P,N,secs_since(_tp));_tp=clk::now();}\
    srand(12345u);                                                                    \
    for(long i=0;i<N;i++){ set_##P##vector_i_##VSFX(x,i,frand());                     \
        for(long j=0;j<N;j++) set_##P##matrix_ij_##MSFX(A,i,j,frand()); }             \
    if(_T){fprintf(stderr,"  [%s N=%ld] fill %.2fs\n",#P,N,secs_since(_tp));_tp=clk::now();}\
    OMP_MV(yc,A,x);                                                                   \
    if(_T){fprintf(stderr,"  [%s N=%ld] cpu-warmup %.2fs\n",#P,N,secs_since(_tp));_tp=clk::now();}\
    double tc=measure_min(g_reps,[&]{ OMP_MV(yc,A,x); },false);                       \
    FAST_MAT(GM,GR,Ad,N); FAST_VEC(GV,GR,xd,N); FAST_VEC(GV,GR,yd,N);                 \
    if(_T){fprintf(stderr,"  [%s N=%ld] dev-alloc %.2fs\n",#P,N,secs_since(_tp));_tp=clk::now();}\
    { GR*ha=(GR*)malloc((size_t)N*N*sizeof(GR));                                      \
      for(long e=0;e<(long)N*N;e++){S*l=(S*)&ha[e]; for(int k=0;k<SZ;k++)l[k]=(S)A->element[k][e];}\
      cudaMemcpy(Ad->element,ha,(size_t)N*N*sizeof(GR),cudaMemcpyHostToDevice); free(ha);}\
    if(_T){fprintf(stderr,"  [%s N=%ld] h2d %.2fs\n",#P,N,secs_since(_tp));_tp=clk::now();}\
    { GR*hx=(GR*)malloc((size_t)N*sizeof(GR));                                        \
      for(long i=0;i<N;i++){S*l=(S*)&hx[i]; for(int k=0;k<SZ;k++)l[k]=(S)x->element[k][i];}\
      cudaMemcpy(xd->element,hx,(size_t)N*sizeof(GR),cudaMemcpyHostToDevice); free(hx);}\
    mul_g##P##matrix_g##P##vec(yd,Ad,xd,g_blocks,g_threads); cudaDeviceSynchronize(); \
    double tg=measure_min(g_reps,[&]{ mul_g##P##matrix_g##P##vec(yd,Ad,xd,g_blocks,g_threads);},true);\
    if(_T){fprintf(stderr,"  [%s N=%ld] kernel(min) %.4fs\n",#P,N,tg);_tp=clk::now();}\
    double re=-1;                                                                     \
    if(N<=g_verify_cap){ GR*hy=(GR*)malloc((size_t)N*sizeof(GR));                     \
        cudaMemcpy(hy,yd->element,(size_t)N*sizeof(GR),cudaMemcpyDeviceToHost);       \
        double mx=0; for(long i=0;i<N;i++){ double a=0,b=0;                           \
          for(int k=0;k<SZ;k++){a+=(double)yc->element[k][i];b+=(double)((S*)&hy[i])[k];}\
          double e=fabs(a-b); if(fabs(a)>1e-300)e/=fabs(a); if(e>mx)mx=e; }           \
        re=mx; free(hy);}                                                             \
    emit("matvec",FAM,#P,BITS,N,0,tg,tc,"omp",re,2.0*(double)N*(double)N);            \
    FAST_FREE(Ad);FAST_FREE(xd);FAST_FREE(yd);                                        \
    free_##P##matrix(A);free_##P##vector(x);free_##P##vector(yc);                     \
}                                                                                    \
static void mm_##P(long N){                                                          \
    M A=init_##P##matrix(N,N),B=init_##P##matrix(N,N),Cc=init_##P##matrix(N,N);       \
    srand(2222u);                                                                     \
    for(long i=0;i<N;i++)for(long j=0;j<N;j++){                                       \
        set_##P##matrix_ij_##MSFX(A,i,j,frand()); set_##P##matrix_ij_##MSFX(B,i,j,frand()); }\
    bool docpu=(N<=g_maxcpu_mm); double tc=0;                                         \
    if(docpu){ OMP_MM(Cc,A,B); tc=measure_min(g_reps,[&]{ OMP_MM(Cc,A,B);},false);}   \
    FAST_MAT(GM,GR,Ad,N); FAST_MAT(GM,GR,Bd,N); FAST_MAT(GM,GR,Cd,N);                 \
    { GR*hb=(GR*)malloc((size_t)N*N*sizeof(GR));                                      \
      for(long e=0;e<(long)N*N;e++){S*l=(S*)&hb[e]; for(int k=0;k<SZ;k++)l[k]=(S)A->element[k][e];}\
      cudaMemcpy(Ad->element,hb,(size_t)N*N*sizeof(GR),cudaMemcpyHostToDevice);       \
      for(long e=0;e<(long)N*N;e++){S*l=(S*)&hb[e]; for(int k=0;k<SZ;k++)l[k]=(S)B->element[k][e];}\
      cudaMemcpy(Bd->element,hb,(size_t)N*N*sizeof(GR),cudaMemcpyHostToDevice); free(hb);}\
    mul_g##P##matrix_dev(Cd,Ad,Bd,g_blocks,g_threads); cudaDeviceSynchronize();       \
    double tg=measure_min(g_reps,[&]{ mul_g##P##matrix_dev(Cd,Ad,Bd,g_blocks,g_threads);},true);\
    double re=-1;                                                                     \
    if(docpu&&N<=g_verify_cap){ GR*hc=(GR*)malloc((size_t)N*N*sizeof(GR));            \
        cudaMemcpy(hc,Cd->element,(size_t)N*N*sizeof(GR),cudaMemcpyDeviceToHost);     \
        double mx=0; for(long e=0;e<(long)N*N;e++){ double a=0,b=0;                   \
          for(int k=0;k<SZ;k++){a+=(double)Cc->element[k][e];b+=(double)((S*)&hc[e])[k];}\
          double ee=fabs(a-b); if(fabs(a)>1e-300)ee/=fabs(a); if(ee>mx)mx=ee; }       \
        re=mx; free(hc);}                                                             \
    emit("matmul",FAM,#P,BITS,N,0,tg,tc,docpu?"omp":"skip",re,2.0*(double)N*(double)N*(double)N);\
    FAST_FREE(Ad);FAST_FREE(Bd);FAST_FREE(Cd);                                        \
    free_##P##matrix(A);free_##P##matrix(B);free_##P##matrix(Cc);                     \
}
DEFINE_DENSE(dd,DDVector,DDMatrix,GDDVector,GDDMatrix,gdd_real,double,DDSIZE,d,d,106,"double-based",_bncomp_mul_ddmatrix_ddvec,_bncomp_mul_ddmatrix)
DEFINE_DENSE(td,TDVector,TDMatrix,GTDVector,GTDMatrix,gtd_real,double,TDSIZE,d,d,159,"double-based",_bncomp_mul_tdmatrix_tdvec,_bncomp_mul_tdmatrix)
DEFINE_DENSE(qd,QDVector,QDMatrix,GQDVector,GQDMatrix,gqd_real,double,QDSIZE,d,d,212,"double-based",_bncomp_mul_qdmatrix_qdvec,_bncomp_mul_qdmatrix)
DEFINE_DENSE(ds,DSVector,DSMatrix,GDSVector,GDSMatrix,gds_real,float, DSSIZE,f,d, 48,"float-based", _bncomp_mul_dsmatrix_dsvec,_bncomp_mul_dsmatrix)
DEFINE_DENSE(ts,TSVector,TSMatrix,GTSVector,GTSMatrix,gts_real,float, TSSIZE,f,f, 72,"float-based", _bncomp_mul_tsmatrix_tsvec,_bncomp_mul_tsmatrix)
DEFINE_DENSE(qs,QSVector,QSMatrix,GQSVector,GQSMatrix,gqs_real,float, QSSIZE,f,f, 96,"float-based", _bncomp_mul_qsmatrix_qsvec,_bncomp_mul_qsmatrix)

static void parse_sizes(const char*s,std::vector<long>&o){ o.clear(); const char*p=s;
    while(*p){char*e; long v=strtol(p,&e,10); if(e==p)break; if(v>0)o.push_back(v); p=e;
        while(*p==','||*p==' ')++p;} }

int main(int argc,char**argv){
    std::vector<long> mv_sizes={512,1024,2048,4096};
    std::vector<long> mm_sizes={256,512,1024,2048};
    int dev=0; const char*out=nullptr; int nt=0; std::string ops="matvec,matmul";
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc)g_reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dev")&&i+1<argc)dev=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--blocks")&&i+1<argc)g_blocks=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)g_threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--maxcpu-mm")&&i+1<argc)g_maxcpu_mm=atol(argv[++i]);
        else if(!strcmp(argv[i],"--verify-cap")&&i+1<argc)g_verify_cap=atol(argv[++i]);
        else if(!strcmp(argv[i],"--threads-cpu")&&i+1<argc)nt=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--ops")&&i+1<argc)ops=argv[++i];
        else if(!strcmp(argv[i],"--out")&&i+1<argc)out=argv[++i];
        else if(!strcmp(argv[i],"--mv-sizes")&&i+1<argc)parse_sizes(argv[++i],mv_sizes);
        else if(!strcmp(argv[i],"--mm-sizes")&&i+1<argc)parse_sizes(argv[++i],mm_sizes);
    }
    g_out=out?fopen(out,"w"):stdout; if(!g_out){perror("out");return 1;}
    if(nt>0) omp_set_num_threads(nt);
    int used_nt=1;
    #pragma omp parallel
    { if(omp_get_thread_num()==0) used_nt=omp_get_num_threads(); }
    unsigned int cw; fpu_fix_start(&cw);
    cudaSetDevice(dev); cudaDeviceProp prop; cudaGetDeviceProperties(&prop,dev);
    fprintf(stderr,"# gpu_omp_bench(dense): %s sm_%d%d | OMP=%d | reps=%d | grid=%dx%d\n",
            prop.name,prop.major,prop.minor,used_nt,g_reps,g_blocks,g_threads);
    /* GxxStart/End only initialise gdtq transcendental constant tables (exp/log/
     * sincos), which basic linear (+,*,matmul) does not use; skipping them saves
     * ~40s of one-time overhead. Correctness is confirmed by relerr. */
    if(getenv("GDTQ_CTX")){ GDDStart(dev);GTDStart(dev);GQDStart(dev);GDSStart(dev);GTSStart(dev);GQSStart(dev); }
    fprintf(g_out,"op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops\n");
    fflush(g_out);
    bool do_mv=ops.find("matvec")!=std::string::npos, do_mm=ops.find("matmul")!=std::string::npos;
    if(do_mv){ fprintf(stderr,"== matvec ==\n");
      for(long N:mv_sizes){ mv_dd(N);mv_td(N);mv_qd(N);mv_ds(N);mv_ts(N);mv_qs(N);
        fprintf(stderr,"  matvec N=%ld done\n",N);} }
    if(do_mm){ fprintf(stderr,"== matmul ==\n");
      for(long N:mm_sizes){ mm_dd(N);mm_td(N);mm_qd(N);mm_ds(N);mm_ts(N);mm_qs(N);
        fprintf(stderr,"  matmul N=%ld done\n",N);} }
    if(getenv("GDTQ_CTX")){ GDDEnd();GTDEnd();GQDEnd();GDSEnd();GTSEnd();GQSEnd(); }
    fpu_fix_end(&cw); if(out)fclose(g_out); return 0;
}

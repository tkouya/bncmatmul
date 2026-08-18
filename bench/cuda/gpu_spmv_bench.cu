/*****************************************************************************
 * gpu_spmv_bench.cu -- SpMV GPU(gdtq) vs CPU-OMP/serial, ONE EFT type per build.
 * Select the type with exactly one of -DBENCH_{DD,TD,QD,DS,TS,QS}. Each type's
 * sparse object (g<P>sparse.o) self-contains the device EFT arithmetic, so only
 * one may be device-linked per binary -> compiled once per type.
 *
 *   GPU  : mul_g<P>spmatrix(y_dev, spm, x_dev, blocks, threads)
 *   CPU  : dd/td/qd -> _bncomp_mul_<P>rsmatrix_<P>vec (OpenMP)
 *          ds/ts/qs -> mul_<P>rsmatrix_<P>vec        (serial; no OMP float SpMV)
 * Synthetic banded matrix, half-bandwidth HBW (=> ~2*HBW+1 nnz/row).
 * CSV row appended: op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops
 *****************************************************************************/
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
#include <omp.h>
#include "gqd_type.h"

/*---- select the type ----*/
#if defined(BENCH_DD)
  #define PT dd
  #define VT DDVector
  #define GR gdd_real
  #define ST double
  #define SZ DDSIZE
  #define SPMT GDDSPMatrix
  #define RST DDRSMatrix
  #define VSFX d
  #define BITS 106
  #define FAM "double-based"
  #define OMPSPMV 1
  #include "gddlinear.h"
  #include "gddsparse.h"
  #include "ddlinear.h"
#elif defined(BENCH_TD)
  #define PT td
  #define VT TDVector
  #define GR gtd_real
  #define ST double
  #define SZ TDSIZE
  #define SPMT GTDSPMatrix
  #define RST TDRSMatrix
  #define VSFX d
  #define BITS 159
  #define FAM "double-based"
  #define OMPSPMV 1
  #include "gtdlinear.h"
  #include "gtdsparse.h"
  #include "tdlinear.h"
#elif defined(BENCH_QD)
  #define PT qd
  #define VT QDVector
  #define GR gqd_real
  #define ST double
  #define SZ QDSIZE
  #define SPMT GQDSPMatrix
  #define RST QDRSMatrix
  #define VSFX d
  #define BITS 212
  #define FAM "double-based"
  #define OMPSPMV 1
  #include "gddlinear.h"
  #include "gqdsparse.h"
  #include "qdlinear.h"
#elif defined(BENCH_DS)
  #define PT ds
  #define VT DSVector
  #define GR gds_real
  #define ST float
  #define SZ DSSIZE
  #define SPMT GDSSPMatrix
  #define RST DSRSMatrix
  #define VSFX d
  #define BITS 48
  #define FAM "float-based"
  #define OMPSPMV 0
  #include "gdslinear.h"
  #include "gdssparse.h"
  #include "dslinear.h"
#elif defined(BENCH_TS)
  #define PT ts
  #define VT TSVector
  #define GR gts_real
  #define ST float
  #define SZ TSSIZE
  #define SPMT GTSSPMatrix
  #define RST TSRSMatrix
  #define VSFX f
  #define BITS 72
  #define FAM "float-based"
  #define OMPSPMV 0
  #include "gtslinear.h"
  #include "gtssparse.h"
  #include "tslinear.h"
#elif defined(BENCH_QS)
  #define PT qs
  #define VT QSVector
  #define GR gqs_real
  #define ST float
  #define SZ QSSIZE
  #define SPMT GQSSPMatrix
  #define RST QSRSMatrix
  #define VSFX f
  #define BITS 96
  #define FAM "float-based"
  #define OMPSPMV 0
  #include "gdslinear.h"
  #include "gqssparse.h"
  #include "qslinear.h"
#else
  #error "define one of BENCH_DD/TD/QD/DS/TS/QS"
#endif

#define USE_DDLINEAR
#define USE_TDLINEAR
#define USE_QDLINEAR
#define USE_DSLINEAR
#define USE_TSLINEAR
#define USE_QSLINEAR
extern "C" {           /* bncsparse.h is not extern-C guarded; the lib exports C names */
#include "bncsparse.h"
}
#include "bncomp.h"

#define CAT_(a,b) a##b
#define CAT(a,b) CAT_(a,b)
#define STR_(x) #x
#define STR(x) STR_(x)

/* name builders */
#define INIT_VEC       CAT(init_,CAT(PT,vector))
#define FREE_VEC       CAT(free_,CAT(PT,vector))
#define SET_VEC_I      CAT(CAT(set_,CAT(PT,vector_i_)),VSFX)
#define INIT_RS        CAT(init_,CAT(PT,rsmatrix))
#define SET_RS_IJ      CAT(set_,CAT(PT,rsmatrix_ij))
#define FREE_RS        CAT(free_,CAT(PT,rsmatrix))
#define SET_NZ_ROW     CAT(set_nzero_row_dim_,PT)
#define GINIT_SPM      CAT(CAT(init_g,PT),spmatrix_dev)
#define GFREE_SPM      CAT(CAT(free_g,PT),spmatrix_dev)
#define GMUL_SPM       CAT(mul_g,CAT(PT,spmatrix))
#define OMP_SPMV_FN    CAT(CAT(_bncomp_mul_,PT),CAT(rsmatrix_,CAT(PT,vec)))
#define SER_SPMV_FN    CAT(CAT(mul_,PT),CAT(rsmatrix_,CAT(PT,vec)))
#define GSTART         CAT(CAT(G,0),0) /* unused placeholder */

using clk=std::chrono::high_resolution_clock;
static double secs(clk::time_point t0){return std::chrono::duration<double>(clk::now()-t0).count();}
static double g_cap=8.0; static int g_reps=3; static long g_verify_cap=1<<20;
static int g_blocks=256,g_threads=256;
template<typename Fn> static double measure_min(int reps,Fn fn,bool gpu){
    double best=1e30; for(int i=0;i<reps;++i){auto t0=clk::now();fn();if(gpu)cudaDeviceSynchronize();
        double s=secs(t0); if(s<best)best=s; if(s>=g_cap)break;} return best; }
static double frand(){ return (double)rand()/(double)RAND_MAX*2.0-1.0; }
static FILE *g_out=nullptr;

static void run_spmv(long N,long HBW){
    long *ncd=(long*)malloc(N*sizeof(long)); long nnz=0;
    for(long i=0;i<N;i++){ long lo=i-HBW<0?0:i-HBW, hi=i+HBW>=N?N-1:i+HBW; ncd[i]=hi-lo+1; nnz+=ncd[i]; }
    RST As=INIT_RS(N,ncd,nnz);
    long *row_ptr=(long*)malloc((N+1)*sizeof(long));
    long *col_idx=(long*)malloc(nnz*sizeof(long));
    GR *val=(GR*)calloc(nnz,sizeof(GR));
    srand(777u); long k=0; row_ptr[0]=0;
    for(long i=0;i<N;i++){ long lo=i-HBW<0?0:i-HBW, hi=i+HBW>=N?N-1:i+HBW; long jj=0;
        for(long j=lo;j<=hi;j++){ double v=frand();
            As->nzero_index[i][jj]=j;   /* set_*rsmatrix_ij only writes an already-recorded index */
            ST vv[SZ]; for(int t=0;t<SZ;t++)vv[t]=(ST)0; vv[0]=(ST)v;
            SET_RS_IJ(As,i,j,vv);
            col_idx[k]=j; ((ST*)&val[k])[0]=(ST)v; k++; jj++; }
        row_ptr[i+1]=k; }
    SET_NZ_ROW(As);
    /* The OMP AVX512 EFT SpMV kernels iterate to real_nzero_col_dim (padded to
     * the SIMD width) and gather nzero_index[i][j+k] in the padding region, but
     * init_*rsmatrix allocates nzero_index[i] only nzero_col_dim long. Extend it
     * to the padded length, filling the gap with a valid (last) column index; the
     * padded element values are zero so the numeric result is unchanged. */
    for(long i=0;i<N;i++){ long real=As->real_nzero_col_dim[i], nz=As->nzero_col_dim[i];
        if(real>nz && nz>0){ As->nzero_index[i]=(long*)realloc(As->nzero_index[i],real*sizeof(long));
            for(long jj=nz;jj<real;jj++) As->nzero_index[i][jj]=As->nzero_index[i][nz-1]; } }
    VT x=INIT_VEC(N), yc=INIT_VEC(N);
    GR *hx=(GR*)calloc(N,sizeof(GR));
    srand(999u); for(long i=0;i<N;i++){ double v=frand(); SET_VEC_I(x,i,v); ((ST*)&hx[i])[0]=(ST)v; }
    const char *cpub;
#if OMPSPMV
    OMP_SPMV_FN(yc,As,x); cpub="omp";
    double tc=measure_min(g_reps,[&]{ OMP_SPMV_FN(yc,As,x);},false);
#else
    SER_SPMV_FN(yc,As,x); cpub="serial";
    double tc=measure_min(g_reps,[&]{ SER_SPMV_FN(yc,As,x);},false);
#endif
    SPMT spm=GINIT_SPM(N,N,nnz,row_ptr,col_idx,val);
    GR *xd,*yd; cudaMalloc(&xd,N*sizeof(GR)); cudaMalloc(&yd,N*sizeof(GR));
    cudaMemcpy(xd,hx,N*sizeof(GR),cudaMemcpyHostToDevice);
    GMUL_SPM(yd,spm,xd,g_blocks,g_threads); cudaDeviceSynchronize();
    double tg=measure_min(g_reps,[&]{ GMUL_SPM(yd,spm,xd,g_blocks,g_threads);},true);
    double re=-1;
    if(N<=g_verify_cap){ GR *hy=(GR*)calloc(N,sizeof(GR));
        cudaMemcpy(hy,yd,N*sizeof(GR),cudaMemcpyDeviceToHost);
        double mx=0; for(long i=0;i<N;i++){ double a=0,b=0;
            for(int t=0;t<SZ;t++){ a+=(double)yc->element[t][i]; b+=(double)((ST*)&hy[i])[t]; }
            double e=fabs(a-b); if(fabs(a)>1e-300)e/=fabs(a); if(e>mx)mx=e; }
        re=mx; free(hy); }
    double gmf=(tg>0)?2.0*nnz/tg/1e6:0, cmf=(tc>0)?2.0*nnz/tc/1e6:0, sp=(tg>0)?tc/tg:0;
    fprintf(g_out,"spmv,%s,%s,%d,%ld,%ld,%.6e,%.6e,%s,%.3f,%.3e,%.4f,%.4f\n",
            FAM,STR(PT),BITS,N,nnz,tg,tc,cpub,sp,re,gmf,cmf);
    fflush(g_out);
    cudaFree(xd);cudaFree(yd); GFREE_SPM(spm);
    free(row_ptr);free(col_idx);free(val);free(hx);free(ncd);
    FREE_RS(As); FREE_VEC(x); FREE_VEC(yc);
}

/* gdtq context start/end for this type */
#if defined(BENCH_DD)
  #define CTX_START GDDStart
  #define CTX_END   GDDEnd
#elif defined(BENCH_TD)
  #define CTX_START GTDStart
  #define CTX_END   GTDEnd
#elif defined(BENCH_QD)
  #define CTX_START GQDStart
  #define CTX_END   GQDEnd
#elif defined(BENCH_DS)
  #define CTX_START GDSStart
  #define CTX_END   GDSEnd
#elif defined(BENCH_TS)
  #define CTX_START GTSStart
  #define CTX_END   GTSEnd
#elif defined(BENCH_QS)
  #define CTX_START GQSStart
  #define CTX_END   GQSEnd
#endif

int main(int argc,char**argv){
    std::vector<long> sizes={4096,16384,65536,262144,1048576};
    long HBW=15; int dev=0; const char*out=nullptr; int nt=0; bool append=false;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc)g_reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--dev")&&i+1<argc)dev=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--band")&&i+1<argc)HBW=atol(argv[++i]);
        else if(!strcmp(argv[i],"--blocks")&&i+1<argc)g_blocks=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)g_threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--verify-cap")&&i+1<argc)g_verify_cap=atol(argv[++i]);
        else if(!strcmp(argv[i],"--threads-cpu")&&i+1<argc)nt=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--out")&&i+1<argc)out=argv[++i];
        else if(!strcmp(argv[i],"--append"))append=true;
        else if(!strcmp(argv[i],"--sizes")&&i+1<argc){ sizes.clear(); char*p=argv[++i];
            while(*p){char*e; long v=strtol(p,&e,10); if(e==p)break; if(v>0)sizes.push_back(v); p=e;
                while(*p==','||*p==' ')++p;} }
    }
    g_out = out? fopen(out,append?"a":"w") : stdout;
    if(!g_out){perror("out");return 1;}
    if(nt>0) omp_set_num_threads(nt);
    unsigned int cw; fpu_fix_start(&cw);
    cudaSetDevice(dev);
    /* SpMV uses only +,* EFT arithmetic; no gdtq constant-table context needed. */
    if(!append && out) fprintf(g_out,"op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops\n");
    for(long N:sizes){ run_spmv(N,HBW); fprintf(stderr,"  spmv %s N=%ld done\n",STR(PT),N); }
    fpu_fix_end(&cw); if(out)fclose(g_out); return 0;
}

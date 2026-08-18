/*****************************************************************
 * cuda_dense_cm.cu -- GPU vs CPU benchmark, COMPLEX multi-component.
 *
 * ONE precision per build (each cg*linear.o embeds the device-math aggregator,
 * so objects of different precisions cannot share a binary).  All type names
 * and entry points are injected from the build script via -D macros:
 *
 *   PREC_NAME  "cdd"            CGHDR        "cgddlinear.h"
 *   MT VT      CDDMatrix/Vector GMT GVT      CGDDMatrix/Vector
 *   BASE SZ    double DDSIZE
 *   INIT_M INIT_V FREE_M FREE_V SET_M SET_V
 *   GINIT_M GINIT_V GFREE_M GFREE_V H2D_M H2D_V D2H_M D2H_V
 *   GMM GMV                      GPU matmul / matvec
 *   NORM2V NORMFM SUBV SUBM      CPU error metric
 *   CALL_CMM(c,a,b) CALL_CMV(v,a,x)   CPU matmul / matvec (omp or serial)
 *   CPUKIND   "omp" | "serial"
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

#include CGHDR              /* the one cg*linear.h for this precision */

#ifdef USE_OMP_CPU
#include <omp.h>
/* OpenMP CPU entry points are not in any header we can include (bncomp.h pulls
 * bncsparse.h -> single/double complex clash); declare the injected names. */
extern "C" void CMM_FN(MT, MT, MT, long int);   /* _bncomp_mul_<p>matrix_strassen */
extern "C" void CMV_FN(VT, MT, VT);             /* _bncomp_mul_<p>matrix_<p>vec_4m */
#endif

static FILE *g_csv = nullptr;
static void emit_csv(const char *op,const char*prec,long dim,long nnz,double tc,double tg,const char*kind,double re){
    if(!g_csv)return; fprintf(g_csv,"RESULT,%s,%s,%ld,%ld,%.9g,%.9g,%s,%.6e\n",op,prec,dim,nnz,tc,tg,kind,re); fflush(g_csv);
}
using clk=std::chrono::high_resolution_clock;
static double secs(clk::time_point t0){return std::chrono::duration<double>(clk::now()-t0).count();}
template<class F> static double tmin_gpu(int r,F f){double b=1e30;for(int i=0;i<r;i++){auto t=clk::now();f();cudaDeviceSynchronize();double s=secs(t);if(s<b)b=s;}return b;}
template<class F> static double tmin_cpu(int r,F f){double b=1e30;for(int i=0;i<r;i++){auto t=clk::now();f();double s=secs(t);if(s<b)b=s;}return b;}

struct Opts{int reps=3,blocks=128,threads=128;long max_cpu=-1;std::vector<long> sizes;};

static void parse_sizes(const char*s,std::vector<long>&o){o.clear();const char*p=s;while(*p){char*e=0;long v=strtol(p,&e,10);if(e==p||v<=0)break;o.push_back(v);p=e;while(*p==','||*p==' ')++p;}}

int main(int argc,char**argv)
{
    Opts o; o.sizes={128,256,512,1024,2048};
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc)o.reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--blocks")&&i+1<argc)o.blocks=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)o.threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-cpu")&&i+1<argc)o.max_cpu=atol(argv[++i]);
        else if(!strcmp(argv[i],"--sizes")&&i+1<argc)parse_sizes(argv[++i],o.sizes);
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
#ifdef USE_OMP_CPU
    const char*env=getenv("OMP_NUM_THREADS");int nt=env?atoi(env):omp_get_max_threads();if(nt<1)nt=omp_get_max_threads();omp_set_num_threads(nt);
#endif
    g_csv=stdout;
    cudaSetDevice(0);cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf(" %s complex dense : device %s sm_%d%d  CPU=%s\n",PREC_NAME,p.name,p.major,p.minor,CPUKIND);

    for(long N:o.sizes){
        bool run_cpu = o.max_cpu<0 || N<=o.max_cpu;
        MT A=INIT_M(N,N),B=INIT_M(N,N),Cs=INIT_M(N,N),Cp=INIT_M(N,N);
        VT x=INIT_V(N),ys=INIT_V(N),yp=INIT_V(N);
        /* real-valued complex inputs (im=0 from init); the complex kernels still
         * perform full complex arithmetic, so timing is representative. */
        srand(20260513u);
        for(long i=0;i<N;i++){ SET_V(x,i,(double)rand()/RAND_MAX*2-1);
            for(long j=0;j<N;j++){ SET_M(A,i,j,(double)rand()/RAND_MAX*2-1);
                                   SET_M(B,i,j,(double)rand()/RAND_MAX*2-1);} }

        GMT Ad=GINIT_M(N,N),Bd=GINIT_M(N,N),Cd=GINIT_M(N,N);
        GVT xd=GINIT_V(N),yd=GINIT_V(N);
        H2D_M(Ad,A);H2D_M(Bd,B);H2D_V(xd,x);

        /* matmul */
        double tc=0,tg=0,re=0;
        GMM(Cd,Ad,Bd,o.blocks,o.threads);cudaDeviceSynchronize();
        tg=tmin_gpu(o.reps,[&]{GMM(Cd,Ad,Bd,o.blocks,o.threads);});
#ifdef SINGLE_BASED
#  define DO_CMM(c,a,b) CMM_FN(c,a,b)          /* serial mul_<p>matrix_4m */
#else
#  define DO_CMM(c,a,b) CMM_FN(c,a,b,16)       /* omp _bncomp_..._strassen */
#endif
        if(run_cpu){ DO_CMM(Cs,A,B); tc=tmin_cpu(o.reps,[&]{DO_CMM(Cs,A,B);});
            D2H_M(Cp,Cd); BASE ns[SZ],nd[SZ]; NORMFM(ns,Cs); SUBM(Cp,Cs,Cp); NORMFM(nd,Cp);
            double a=(double)ns[0],b=(double)nd[0]; re=a!=0?b/a:b; }
        emit_csv("matmul",PREC_NAME,N,0,tc,tg,run_cpu?CPUKIND:"none",run_cpu?re:-1);

        /* matvec */
        tc=tg=re=0;
        GMV(yd,Ad,xd,o.blocks,o.threads);cudaDeviceSynchronize();
        tg=tmin_gpu(o.reps,[&]{GMV(yd,Ad,xd,o.blocks,o.threads);});
        if(run_cpu){ CMV_FN(ys,A,x); tc=tmin_cpu(o.reps,[&]{CMV_FN(ys,A,x);});
            D2H_V(yp,yd); BASE ns[SZ],nd[SZ]; NORM2V(ns,ys); SUBV(yp,ys,yp); NORM2V(nd,yp);
            double a=(double)ns[0],b=(double)nd[0]; re=a!=0?b/a:b; }
        emit_csv("matvec",PREC_NAME,N,0,tc,tg,run_cpu?CPUKIND:"none",run_cpu?re:-1);

        GFREE_M(Ad);GFREE_M(Bd);GFREE_M(Cd);GFREE_V(xd);GFREE_V(yd);
        FREE_M(A);FREE_M(B);FREE_M(Cs);FREE_M(Cp);FREE_V(x);FREE_V(ys);FREE_V(yp);
    }
    return 0;
}

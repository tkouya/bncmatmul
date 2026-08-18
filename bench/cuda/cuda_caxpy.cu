/*****************************************************************
 * cuda_caxpy.cu -- GPU complex axpy-class benchmark (c = val * a, the
 * cmul vector op; memory-bound).  ONE complex precision per build, all type
 * names injected by the build script via -D macros.  GPU-only timing (the
 * complex cmul has no uniform CPU-OMP counterpart across all precisions), so
 * only a device vector is needed -- no host data / correctness check here
 * (correctness of the complex kernels is covered by the matmul/matvec drivers).
 *
 *   PREC_NAME  "ccdd" etc.   CGHDR  "cg<p>linear.h" or "g<p>linear.h"
 *   GVT        GPU vector type      GINIT_V / GFREE_V   init/free on device
 *   GCMUL      cmul_*vector_dev     signature (c, re, im, a, nbg, ntb)
 *   MULTICOMP  defined -> scalar is GR (gXX_real, limb array); else BASE scalar
 *   GR BASE SZ
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

#include CGHDR

static FILE *g_csv = nullptr;
static void emit_csv(const char *op,const char*prec,long dim,long nnz,double tc,double tg,const char*kind,double re){
    if(!g_csv)return; fprintf(g_csv,"RESULT,%s,%s,%ld,%ld,%.9g,%.9g,%s,%.6e\n",op,prec,dim,nnz,tc,tg,kind,re); fflush(g_csv);
}
using clk=std::chrono::high_resolution_clock;
static double secs(clk::time_point t0){return std::chrono::duration<double>(clk::now()-t0).count();}
template<class F> static double tmin_gpu(int r,F f){double b=1e30;for(int i=0;i<r;i++){auto t=clk::now();f();cudaDeviceSynchronize();double s=secs(t);if(s<b)b=s;}return b;}

struct Opts{int reps=3,blocks=128,threads=128;long max_cpu=-1;std::vector<long> sizes;};
static void parse_sizes(const char*s,std::vector<long>&o){o.clear();const char*p=s;while(*p){char*e=0;long v=strtol(p,&e,10);if(e==p||v<=0)break;o.push_back(v);p=e;while(*p==','||*p==' ')++p;}}

int main(int argc,char**argv)
{
    Opts o; o.sizes={4096,16384,65536,262144,1048576};
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc)o.reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--blocks")&&i+1<argc)o.blocks=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)o.threads=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--max-cpu")&&i+1<argc)o.max_cpu=atol(argv[++i]); /* ignored (GPU-only) */
        else if(!strcmp(argv[i],"--sizes")&&i+1<argc)parse_sizes(argv[++i],o.sizes);
        else if(!strcmp(argv[i],"--axpy-sizes")&&i+1<argc)parse_sizes(argv[++i],o.sizes);
        else { fprintf(stderr,"unknown arg %s\n",argv[i]); return 1; }
    }
    g_csv=stdout;
    cudaSetDevice(0);cudaDeviceProp p;cudaGetDeviceProperties(&p,0);
    printf(" %s complex axpy(cmul) : device %s sm_%d%d  (GPU-only)\n",PREC_NAME,p.name,p.major,p.minor);

    /* scalar val = 0.7283105 + 0.1300000 i */
#ifdef MULTICOMP
    GR vre, vim; memset(&vre,0,sizeof(vre)); memset(&vim,0,sizeof(vim));
    { BASE *r=reinterpret_cast<BASE*>(&vre), *m=reinterpret_cast<BASE*>(&vim);
      r[0]=(BASE)0.7283105; m[0]=(BASE)0.13; }
#else
    BASE vre=(BASE)0.7283105, vim=(BASE)0.13;
#endif

    for(long N:o.sizes){
        GVT ad=GINIT_V(N), cd=GINIT_V(N);
        GCMUL(cd, vre, vim, ad, o.blocks, o.threads); cudaDeviceSynchronize();
        if(cudaGetLastError()!=cudaSuccess){ GFREE_V(ad);GFREE_V(cd); continue; }
        double tg=tmin_gpu(o.reps,[&]{GCMUL(cd, vre, vim, ad, o.blocks, o.threads);});
        emit_csv("axpy",PREC_NAME,N,0,0,tg,"none",-1);
        GFREE_V(ad);GFREE_V(cd);
    }
    return 0;
}

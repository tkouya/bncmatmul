/*****************************************************************************
 * cmpf_bench.cu -- COMPLEX multiple-precision (MPC) basic linear algebra,
 * GPU (mpc_cuda device MPC) vs CPU OpenMP-parallel (same mpc_cuda host MPC),
 * at a fixed precision PREC (bits per real/imag component).
 *
 *   axpy   : y = a + alpha*x                (vectors, complex)
 *   matvec : y = A*x
 *   matmul : C = A*B
 * All accumulation at MPC precision PREC.  The element routines are
 * __host__ __device__, so the CPU baseline runs the IDENTICAL arithmetic,
 * parallelised with OpenMP (num_threads = OMP_NUM_THREADS) -- an
 * apples-to-apples "same-precision parallel CPU routine".
 *
 * Build per precision: -DPREC=<bits>.  See build_gpu_omp.sh (cmpf section).
 * CSV: op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"
#include <omp.h>

#ifndef PREC
#define PREC 212
#endif
#ifndef SLAB
#define SLAB (64*1024)
#endif

#define NLIMBS ((PREC + 8*(int)sizeof(cu_mp_limb_t) - 1) / (8*(int)sizeof(cu_mp_limb_t)))
#define CU_MPC_DECL_INIT(z)                                                   \
  cu_mpc_t z; cu_mp_limb_t z##_rl[NLIMBS], z##_il[NLIMBS];                    \
  cu_mpfr_custom_init_set(cu_mpc_realref(z), CU_MPFR_ZERO_KIND, 0, PREC, z##_rl); \
  cu_mpfr_custom_init_set(cu_mpc_imagref(z), CU_MPFR_ZERO_KIND, 0, PREC, z##_il)

/*------------------ element routines (host + device) ------------------*/
__host__ __device__ static void
cdot_row(const double *Ar,const double *Ai,const double *xr,const double *xi,
         int n,double *yr,double *yi){
  CU_MPC_DECL_INIT(acc);CU_MPC_DECL_INIT(a);CU_MPC_DECL_INIT(xx);CU_MPC_DECL_INIT(t);
  for(int j=0;j<n;++j){
#ifdef __CUDA_ARCH__
    mpc_cuda_arena_reset();
#endif
    cu_mpc_set_d_d(a, Ar[j],Ai[j],CU_MPC_RNDNN);
    cu_mpc_set_d_d(xx,xr[j],xi[j],CU_MPC_RNDNN);
    cu_mpc_mul(t,a,xx,CU_MPC_RNDNN);
    cu_mpc_add(acc,acc,t,CU_MPC_RNDNN);
  }
  *yr=cu_mpfr_get_d(cu_mpc_realref(acc),CU_MPFR_RNDN);
  *yi=cu_mpfr_get_d(cu_mpc_imagref(acc),CU_MPFR_RNDN);
}
__host__ __device__ static void
cdot_ij(const double *Ar,const double *Ai,const double *Br,const double *Bi,
        int n,int i,int j,double *cr,double *ci){
  CU_MPC_DECL_INIT(acc);CU_MPC_DECL_INIT(a);CU_MPC_DECL_INIT(b);CU_MPC_DECL_INIT(t);
  for(int k=0;k<n;++k){
#ifdef __CUDA_ARCH__
    mpc_cuda_arena_reset();
#endif
    size_t ia=(size_t)i*n+k, ib=(size_t)k*n+j;
    cu_mpc_set_d_d(a, Ar[ia],Ai[ia],CU_MPC_RNDNN);
    cu_mpc_set_d_d(b, Br[ib],Bi[ib],CU_MPC_RNDNN);
    cu_mpc_mul(t,a,b,CU_MPC_RNDNN);
    cu_mpc_add(acc,acc,t,CU_MPC_RNDNN);
  }
  *cr=cu_mpfr_get_d(cu_mpc_realref(acc),CU_MPFR_RNDN);
  *ci=cu_mpfr_get_d(cu_mpc_imagref(acc),CU_MPFR_RNDN);
}
__host__ __device__ static void
caxpy_i(const double *ar,const double *ai,double alr,double ali,
        const double *xr,const double *xi,int i,double *yr,double *yi){
#ifdef __CUDA_ARCH__
  mpc_cuda_arena_reset();
#endif
  CU_MPC_DECL_INIT(a);CU_MPC_DECL_INIT(al);CU_MPC_DECL_INIT(x);CU_MPC_DECL_INIT(t);
  cu_mpc_set_d_d(a, ar[i],ai[i],CU_MPC_RNDNN);
  cu_mpc_set_d_d(al,alr,ali,CU_MPC_RNDNN);
  cu_mpc_set_d_d(x, xr[i],xi[i],CU_MPC_RNDNN);
  cu_mpc_mul(t,al,x,CU_MPC_RNDNN);
  cu_mpc_add(t,a,t,CU_MPC_RNDNN);
  *yr=cu_mpfr_get_d(cu_mpc_realref(t),CU_MPFR_RNDN);
  *yi=cu_mpfr_get_d(cu_mpc_imagref(t),CU_MPFR_RNDN);
}

/*------------------ device kernels ------------------*/
__global__ void k_matvec(int n,const double*Ar,const double*Ai,const double*xr,const double*xi,double*yr,double*yi){
  int st=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=st)
    cdot_row(Ar+(size_t)i*n,Ai+(size_t)i*n,xr,xi,n,&yr[i],&yi[i]);
}
__global__ void k_matmul(int n,const double*Ar,const double*Ai,const double*Br,const double*Bi,double*Cr,double*Ci){
  int st=gridDim.x*blockDim.x; long tot=(long)n*n;
  for(long e=blockIdx.x*blockDim.x+threadIdx.x;e<tot;e+=st)
    cdot_ij(Ar,Ai,Br,Bi,n,(int)(e/n),(int)(e%n),&Cr[e],&Ci[e]);
}
__global__ void k_axpy(int n,const double*ar,const double*ai,double alr,double ali,const double*xr,const double*xi,double*yr,double*yi){
  int st=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=st)
    caxpy_i(ar,ai,alr,ali,xr,xi,i,&yr[i],&yi[i]);
}

/*------------------ helpers ------------------*/
using clk=std::chrono::high_resolution_clock;
static double secs(clk::time_point t0){return std::chrono::duration<double>(clk::now()-t0).count();}
static int LB=128, LT=32, g_reps=2;
static char *g_arena=nullptr; static size_t *g_top=nullptr; static size_t g_ntot=0;
static void arena_setup(){
  size_t ntot=(size_t)LB*LT;
  if(ntot!=g_ntot){ if(g_arena)cudaFree(g_arena); if(g_top)cudaFree(g_top);
    cudaMalloc(&g_arena,ntot*(size_t)SLAB); cudaMalloc(&g_top,ntot*sizeof(size_t)); g_ntot=ntot; }
  cudaMemset(g_top,0,ntot*sizeof(size_t));
  mpc_cuda_arena_base=g_arena; mpc_cuda_arena_slab=SLAB; mpc_cuda_arena_top=g_top;
  cudaDeviceSynchronize();
}
static double frand(){ return (double)rand()/(double)RAND_MAX*2.0-1.0; }
static FILE*g_out=nullptr;
static void emit(const char*op,long N,long nnz,double tg,double tc,double relerr,double flop){
  double sp=(tg>0)?tc/tg:0, gmf=(tg>0)?flop/tg/1e6:0, cmf=(tc>0)?flop/tc/1e6:0;
  fprintf(g_out,"%s,complex-mpc,cmpf%d,%d,%ld,%ld,%.6e,%.6e,omp,%.3f,%.3e,%.4f,%.4f\n",
          op,PREC,PREC,N,nnz,tg,tc,sp,relerr,gmf,cmf);
  fflush(g_out);
}

static void bench_matvec(int N){
  size_t MA=(size_t)N*N*sizeof(double), MV=(size_t)N*sizeof(double);
  double*Ar=(double*)malloc(MA),*Ai=(double*)malloc(MA),*xr=(double*)malloc(MV),*xi=(double*)malloc(MV);
  double*ygr=(double*)malloc(MV),*ygi=(double*)malloc(MV),*ycr=(double*)malloc(MV),*yci=(double*)malloc(MV);
  srand(11); for(int i=0;i<N;++i){xr[i]=frand();xi[i]=frand();for(int j=0;j<N;++j){Ar[(size_t)i*N+j]=frand();Ai[(size_t)i*N+j]=frand();}}
  double*dAr,*dAi,*dxr,*dxi,*dyr,*dyi;
  cudaMalloc(&dAr,MA);cudaMalloc(&dAi,MA);cudaMalloc(&dxr,MV);cudaMalloc(&dxi,MV);cudaMalloc(&dyr,MV);cudaMalloc(&dyi,MV);
  cudaMemcpy(dAr,Ar,MA,cudaMemcpyHostToDevice);cudaMemcpy(dAi,Ai,MA,cudaMemcpyHostToDevice);
  cudaMemcpy(dxr,xr,MV,cudaMemcpyHostToDevice);cudaMemcpy(dxi,xi,MV,cudaMemcpyHostToDevice);
  arena_setup();
  k_matvec<<<LB,LT>>>(N,dAr,dAi,dxr,dxi,dyr,dyi); cudaDeviceSynchronize();
  double tg=1e30; for(int r=0;r<g_reps;++r){auto t0=clk::now();k_matvec<<<LB,LT>>>(N,dAr,dAi,dxr,dxi,dyr,dyi);cudaDeviceSynchronize();double s=secs(t0);if(s<tg)tg=s;}
  cudaMemcpy(ygr,dyr,MV,cudaMemcpyDeviceToHost);cudaMemcpy(ygi,dyi,MV,cudaMemcpyDeviceToHost);
  double tc=1e30; for(int r=0;r<g_reps;++r){auto t0=clk::now();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i) cdot_row(Ar+(size_t)i*N,Ai+(size_t)i*N,xr,xi,N,&ycr[i],&yci[i]);
    double s=secs(t0); if(s<tc)tc=s; }
  double mx=0; for(int i=0;i<N;++i){double dr=ycr[i]!=0?fabs((ygr[i]-ycr[i])/ycr[i]):fabs(ygr[i]);double di=yci[i]!=0?fabs((ygi[i]-yci[i])/yci[i]):fabs(ygi[i]);double r=dr>di?dr:di;if(r>mx)mx=r;}
  emit("matvec",N,0,tg,tc,mx,2.0*(double)N*(double)N);
  cudaFree(dAr);cudaFree(dAi);cudaFree(dxr);cudaFree(dxi);cudaFree(dyr);cudaFree(dyi);
  free(Ar);free(Ai);free(xr);free(xi);free(ygr);free(ygi);free(ycr);free(yci);
}
static void bench_matmul(int N){
  size_t MA=(size_t)N*N*sizeof(double);
  double*Ar=(double*)malloc(MA),*Ai=(double*)malloc(MA),*Br=(double*)malloc(MA),*Bi=(double*)malloc(MA);
  double*Cgr=(double*)malloc(MA),*Cgi=(double*)malloc(MA),*Ccr=(double*)malloc(MA),*Cci=(double*)malloc(MA);
  srand(22); for(long e=0;e<(long)N*N;++e){Ar[e]=frand();Ai[e]=frand();Br[e]=frand();Bi[e]=frand();}
  double*dAr,*dAi,*dBr,*dBi,*dCr,*dCi;
  cudaMalloc(&dAr,MA);cudaMalloc(&dAi,MA);cudaMalloc(&dBr,MA);cudaMalloc(&dBi,MA);cudaMalloc(&dCr,MA);cudaMalloc(&dCi,MA);
  cudaMemcpy(dAr,Ar,MA,cudaMemcpyHostToDevice);cudaMemcpy(dAi,Ai,MA,cudaMemcpyHostToDevice);
  cudaMemcpy(dBr,Br,MA,cudaMemcpyHostToDevice);cudaMemcpy(dBi,Bi,MA,cudaMemcpyHostToDevice);
  arena_setup();
  k_matmul<<<LB,LT>>>(N,dAr,dAi,dBr,dBi,dCr,dCi); cudaDeviceSynchronize();
  double tg=1e30; for(int r=0;r<g_reps;++r){auto t0=clk::now();k_matmul<<<LB,LT>>>(N,dAr,dAi,dBr,dBi,dCr,dCi);cudaDeviceSynchronize();double s=secs(t0);if(s<tg)tg=s;}
  cudaMemcpy(Cgr,dCr,MA,cudaMemcpyDeviceToHost);cudaMemcpy(Cgi,dCi,MA,cudaMemcpyDeviceToHost);
  double tc=1e30; for(int r=0;r<g_reps;++r){auto t0=clk::now();
    #pragma omp parallel for schedule(static)
    for(long e=0;e<(long)N*N;++e) cdot_ij(Ar,Ai,Br,Bi,N,(int)(e/N),(int)(e%N),&Ccr[e],&Cci[e]);
    double s=secs(t0); if(s<tc)tc=s; }
  double mx=0; for(long e=0;e<(long)N*N;++e){double dr=Ccr[e]!=0?fabs((Cgr[e]-Ccr[e])/Ccr[e]):fabs(Cgr[e]);double di=Cci[e]!=0?fabs((Cgi[e]-Cci[e])/Cci[e]):fabs(Cgi[e]);double r=dr>di?dr:di;if(r>mx)mx=r;}
  emit("matmul",N,0,tg,tc,mx,2.0*(double)N*(double)N*(double)N);
  cudaFree(dAr);cudaFree(dAi);cudaFree(dBr);cudaFree(dBi);cudaFree(dCr);cudaFree(dCi);
  free(Ar);free(Ai);free(Br);free(Bi);free(Cgr);free(Cgi);free(Ccr);free(Cci);
}
static void bench_axpy(int N){
  size_t MV=(size_t)N*sizeof(double);
  double*ar=(double*)malloc(MV),*ai=(double*)malloc(MV),*xr=(double*)malloc(MV),*xi=(double*)malloc(MV);
  double*ygr=(double*)malloc(MV),*ygi=(double*)malloc(MV),*ycr=(double*)malloc(MV),*yci=(double*)malloc(MV);
  srand(33); for(int i=0;i<N;++i){ar[i]=frand();ai[i]=frand();xr[i]=frand();xi[i]=frand();}
  double alr=1.5, ali=-0.5;
  double*dar,*dai,*dxr,*dxi,*dyr,*dyi;
  cudaMalloc(&dar,MV);cudaMalloc(&dai,MV);cudaMalloc(&dxr,MV);cudaMalloc(&dxi,MV);cudaMalloc(&dyr,MV);cudaMalloc(&dyi,MV);
  cudaMemcpy(dar,ar,MV,cudaMemcpyHostToDevice);cudaMemcpy(dai,ai,MV,cudaMemcpyHostToDevice);
  cudaMemcpy(dxr,xr,MV,cudaMemcpyHostToDevice);cudaMemcpy(dxi,xi,MV,cudaMemcpyHostToDevice);
  int lb=(N+LT-1)/LT; if(lb<1)lb=1;
  arena_setup(); /* arena sized by LB*LT; axpy uses its own grid but SLAB per (LB*LT) threads is enough for 1 madd */
  size_t ntot=(size_t)lb*LT; if(ntot>g_ntot){ cudaFree(g_arena);cudaFree(g_top);
    cudaMalloc(&g_arena,ntot*(size_t)SLAB);cudaMalloc(&g_top,ntot*sizeof(size_t));g_ntot=ntot;
    cudaMemset(g_top,0,ntot*sizeof(size_t)); mpc_cuda_arena_base=g_arena;mpc_cuda_arena_slab=SLAB;mpc_cuda_arena_top=g_top;cudaDeviceSynchronize(); }
  k_axpy<<<lb,LT>>>(N,dar,dai,alr,ali,dxr,dxi,dyr,dyi); cudaDeviceSynchronize();
  double tg=1e30; for(int r=0;r<g_reps;++r){auto t0=clk::now();k_axpy<<<lb,LT>>>(N,dar,dai,alr,ali,dxr,dxi,dyr,dyi);cudaDeviceSynchronize();double s=secs(t0);if(s<tg)tg=s;}
  cudaMemcpy(ygr,dyr,MV,cudaMemcpyDeviceToHost);cudaMemcpy(ygi,dyi,MV,cudaMemcpyDeviceToHost);
  double tc=1e30; for(int r=0;r<g_reps;++r){auto t0=clk::now();
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;++i) caxpy_i(ar,ai,alr,ali,xr,xi,i,&ycr[i],&yci[i]);
    double s=secs(t0); if(s<tc)tc=s; }
  double mx=0; for(int i=0;i<N;++i){double dr=ycr[i]!=0?fabs((ygr[i]-ycr[i])/ycr[i]):fabs(ygr[i]);double di=yci[i]!=0?fabs((ygi[i]-yci[i])/yci[i]):fabs(ygi[i]);double r=dr>di?dr:di;if(r>mx)mx=r;}
  emit("axpy",N,0,tg,tc,mx,2.0*(double)N);
  cudaFree(dar);cudaFree(dai);cudaFree(dxr);cudaFree(dxi);cudaFree(dyr);cudaFree(dyi);
  free(ar);free(ai);free(xr);free(xi);free(ygr);free(ygi);free(ycr);free(yci);
}

int main(int argc,char**argv){
  std::vector<int> mv={128,256,512}, mm={64,128,256}, ax={4096,16384,65536};
  const char*out=nullptr; bool append=false; int dev=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--reps")&&i+1<argc)g_reps=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--lb")&&i+1<argc)LB=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--lt")&&i+1<argc)LT=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dev")&&i+1<argc)dev=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--out")&&i+1<argc)out=argv[++i];
    else if(!strcmp(argv[i],"--append"))append=true;
  }
  cudaSetDevice(dev);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)128*1024*1024);
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)200*1024);
  g_out = out? fopen(out,append?"a":"w") : stdout;
  if(!append && out) fprintf(g_out,"op,family,type,bits,N,nnz,gpu_sec,cpu_sec,cpu_backend,speedup,relerr,gpu_mflops,cpu_mflops\n");
  int nt=1;
  #pragma omp parallel
  { if(omp_get_thread_num()==0) nt=omp_get_num_threads(); }
  fprintf(stderr,"# cmpf_bench PREC=%d OMP=%d reps=%d grid=%dx%d\n",PREC,nt,g_reps,LB,LT);
  for(int N:ax){ bench_axpy(N); fprintf(stderr,"  axpy N=%d done\n",N); }
  for(int N:mv){ bench_matvec(N); fprintf(stderr,"  matvec N=%d done\n",N); }
  for(int N:mm){ bench_matmul(N); fprintf(stderr,"  matmul N=%d done\n",N); }
  if(out)fclose(g_out);
  return 0;
}

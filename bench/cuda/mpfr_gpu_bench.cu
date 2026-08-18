/*****************************************************************************
 * mpfr_gpu_bench.cu -- REAL multiple-precision (MPFR-CUDA) basic linear
 * algebra on the GPU, at a fixed precision PREC (bits per number), PLUS a
 * native double / float reference using the identical kernel structure.
 *
 *   axpy   : y = a + alpha*x           (vectors)            flop = 2 N
 *   matvec : y = A*x                                        flop = 2 N^2
 *   matmul : C = A*B                                        flop = 2 N^3
 *
 * This is the REAL counterpart of cmpf_bench.cu (which is complex MPC): the
 * element routines accumulate with cu_mpfr at PREC bits, one thread per output
 * element / row, so the ONLY difference vs the native double/float kernels is
 * the arithmetic precision -- an apples-to-apples "arbitrary-precision on GPU"
 * baseline for the fixed-precision EFT (gdtq) routines.
 *
 * Build per precision: -DPREC=<bits>  (see build_mpfr_gpu.sh).
 * CSV: op,kind,type,bits,N,gpu_sec,gpu_mflops,relerr
 *****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>
typedef long int gmp_randstate_t[1];
#include "mpc_cuda.cuh"

#ifndef PREC
#define PREC 106
#endif
#ifndef SLAB
#define SLAB (64*1024)
#endif

#define NLIMBS ((PREC + 8*(int)sizeof(cu_mp_limb_t) - 1) / (8*(int)sizeof(cu_mp_limb_t)))
#define CU_MPFR_DECL_INIT(z)                                                   \
  cu_mpfr_t z; cu_mp_limb_t z##_l[NLIMBS];                                     \
  cu_mpfr_custom_init_set(z, CU_MPFR_ZERO_KIND, 0, PREC, z##_l)

/*------------------ MPFR element routines (host + device) ------------------*/
__host__ __device__ static void
rdot_row(const double *A,const double *x,int n,double *y){
  CU_MPFR_DECL_INIT(acc);CU_MPFR_DECL_INIT(a);CU_MPFR_DECL_INIT(xx);CU_MPFR_DECL_INIT(t);
  for(int j=0;j<n;++j){
#ifdef __CUDA_ARCH__
    mpc_cuda_arena_reset();
#endif
    cu_mpfr_set_d(a, A[j], CU_MPFR_RNDN);
    cu_mpfr_set_d(xx,x[j], CU_MPFR_RNDN);
    cu_mpfr_mul(t,a,xx,CU_MPFR_RNDN);
    cu_mpfr_add(acc,acc,t,CU_MPFR_RNDN);
  }
  *y=cu_mpfr_get_d(acc,CU_MPFR_RNDN);
}
__host__ __device__ static void
rdot_ij(const double *A,const double *B,int n,int i,int j,double *c){
  CU_MPFR_DECL_INIT(acc);CU_MPFR_DECL_INIT(a);CU_MPFR_DECL_INIT(b);CU_MPFR_DECL_INIT(t);
  for(int k=0;k<n;++k){
#ifdef __CUDA_ARCH__
    mpc_cuda_arena_reset();
#endif
    size_t ia=(size_t)i*n+k, ib=(size_t)k*n+j;
    cu_mpfr_set_d(a,A[ia],CU_MPFR_RNDN);
    cu_mpfr_set_d(b,B[ib],CU_MPFR_RNDN);
    cu_mpfr_mul(t,a,b,CU_MPFR_RNDN);
    cu_mpfr_add(acc,acc,t,CU_MPFR_RNDN);
  }
  *c=cu_mpfr_get_d(acc,CU_MPFR_RNDN);
}
__host__ __device__ static void
raxpy_i(const double *a,double al,const double *x,int i,double *y){
#ifdef __CUDA_ARCH__
  mpc_cuda_arena_reset();
#endif
  CU_MPFR_DECL_INIT(aa);CU_MPFR_DECL_INIT(alp);CU_MPFR_DECL_INIT(xx);CU_MPFR_DECL_INIT(t);
  cu_mpfr_set_d(aa, a[i],CU_MPFR_RNDN);
  cu_mpfr_set_d(alp,al,   CU_MPFR_RNDN);
  cu_mpfr_set_d(xx, x[i],CU_MPFR_RNDN);
  cu_mpfr_mul(t,alp,xx,CU_MPFR_RNDN);
  cu_mpfr_add(t,aa,t,CU_MPFR_RNDN);
  *y=cu_mpfr_get_d(t,CU_MPFR_RNDN);
}

/*------------------ MPFR device kernels ------------------*/
__global__ void km_matvec(int n,const double*A,const double*x,double*y){
  int st=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=st) rdot_row(A+(size_t)i*n,x,n,&y[i]);
}
__global__ void km_matmul(int n,const double*A,const double*B,double*C){
  int st=gridDim.x*blockDim.x; long tot=(long)n*n;
  for(long e=blockIdx.x*blockDim.x+threadIdx.x;e<tot;e+=st) rdot_ij(A,B,n,(int)(e/n),(int)(e%n),&C[e]);
}
__global__ void km_axpy(int n,const double*a,double al,const double*x,double*y){
  int st=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=st) raxpy_i(a,al,x,i,&y[i]);
}

/*------------------ native double/float kernels (same structure) ------------------*/
template<class T> __global__ void kn_matvec(int n,const T*A,const T*x,T*y){
  int st=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=st){ T s=0; const T*Ar=A+(size_t)i*n;
    for(int j=0;j<n;++j) s+=Ar[j]*x[j]; y[i]=s; }
}
template<class T> __global__ void kn_matmul(int n,const T*A,const T*B,T*C){
  int st=gridDim.x*blockDim.x; long tot=(long)n*n;
  for(long e=blockIdx.x*blockDim.x+threadIdx.x;e<tot;e+=st){ int i=(int)(e/n),j=(int)(e%n);
    T s=0; for(int k=0;k<n;++k) s+=A[(size_t)i*n+k]*B[(size_t)k*n+j]; C[e]=s; }
}
template<class T> __global__ void kn_axpy(int n,const T*a,T al,const T*x,T*y){
  int st=gridDim.x*blockDim.x;
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=st) y[i]=a[i]+al*x[i];
}

/*------------------ helpers ------------------*/
using clk=std::chrono::high_resolution_clock;
static double secs(clk::time_point t0){return std::chrono::duration<double>(clk::now()-t0).count();}
static int LB=256, LT=128, g_reps=3;
static char *g_arena=nullptr; static size_t *g_top=nullptr; static size_t g_ntot=0;
static void arena_setup(int lb,int lt){
  size_t ntot=(size_t)lb*lt;
  if(ntot>g_ntot){ if(g_arena)cudaFree(g_arena); if(g_top)cudaFree(g_top);
    cudaMalloc(&g_arena,ntot*(size_t)SLAB); cudaMalloc(&g_top,ntot*sizeof(size_t)); g_ntot=ntot; }
  cudaMemset(g_top,0,g_ntot*sizeof(size_t));
  mpc_cuda_arena_base=g_arena; mpc_cuda_arena_slab=SLAB; mpc_cuda_arena_top=g_top;
  cudaDeviceSynchronize();
}
static double frand(){ return (double)rand()/(double)RAND_MAX*2.0-1.0; }
static FILE*g_out=nullptr;
static void emit(const char*op,const char*kind,const char*type,int bits,long N,double tg,double relerr,double flop){
  double gmf=(tg>0)?flop/tg/1e6:0;
  fprintf(g_out,"%s,%s,%s,%d,%ld,%.6e,%.4f,%.3e\n",op,kind,type,bits,N,tg,gmf,relerr);
  fflush(g_out);
}
template<typename F> static double tmin(F f){ double b=1e30; for(int r=0;r<g_reps;++r){auto t0=clk::now(); f(); cudaDeviceSynchronize(); double s=secs(t0); if(s<b)b=s;} return b; }

/*==================== MPFR benches ====================*/
static void mpfr_axpy(int N){
  size_t MV=(size_t)N*sizeof(double);
  double*a=(double*)malloc(MV),*x=(double*)malloc(MV),*yg=(double*)malloc(MV);
  srand(33); for(int i=0;i<N;++i){a[i]=frand();x[i]=frand();}
  double al=1.5;
  double*da,*dx,*dy; cudaMalloc(&da,MV);cudaMalloc(&dx,MV);cudaMalloc(&dy,MV);
  cudaMemcpy(da,a,MV,cudaMemcpyHostToDevice);cudaMemcpy(dx,x,MV,cudaMemcpyHostToDevice);
  int lb=(N+LT-1)/LT; if(lb<1)lb=1; arena_setup(lb,LT);
  km_axpy<<<lb,LT>>>(N,da,al,dx,dy); cudaDeviceSynchronize();
  double tg=tmin([&]{km_axpy<<<lb,LT>>>(N,da,al,dx,dy);});
  cudaMemcpy(yg,dy,MV,cudaMemcpyDeviceToHost);
  double mx=0; for(int i=0;i<N;++i){double ref=a[i]+al*x[i]; double e=ref!=0?fabs((yg[i]-ref)/ref):fabs(yg[i]); if(e>mx)mx=e;}
  emit("axpy","mpfr","mpfr",PREC,N,tg,mx,2.0*(double)N);
  cudaFree(da);cudaFree(dx);cudaFree(dy); free(a);free(x);free(yg);
}
static void mpfr_matvec(int N){
  size_t MA=(size_t)N*N*sizeof(double), MV=(size_t)N*sizeof(double);
  double*A=(double*)malloc(MA),*x=(double*)malloc(MV),*yg=(double*)malloc(MV);
  srand(11); for(int i=0;i<N;++i){x[i]=frand();for(int j=0;j<N;++j)A[(size_t)i*N+j]=frand();}
  double*dA,*dx,*dy; cudaMalloc(&dA,MA);cudaMalloc(&dx,MV);cudaMalloc(&dy,MV);
  cudaMemcpy(dA,A,MA,cudaMemcpyHostToDevice);cudaMemcpy(dx,x,MV,cudaMemcpyHostToDevice);
  arena_setup(LB,LT);
  km_matvec<<<LB,LT>>>(N,dA,dx,dy); cudaDeviceSynchronize();
  double tg=tmin([&]{km_matvec<<<LB,LT>>>(N,dA,dx,dy);});
  cudaMemcpy(yg,dy,MV,cudaMemcpyDeviceToHost);
  double mx=0; for(int i=0;i<N;++i){double ref=0; for(int j=0;j<N;++j)ref+=A[(size_t)i*N+j]*x[j]; double e=ref!=0?fabs((yg[i]-ref)/ref):fabs(yg[i]); if(e>mx)mx=e;}
  emit("matvec","mpfr","mpfr",PREC,N,tg,mx,2.0*(double)N*(double)N);
  cudaFree(dA);cudaFree(dx);cudaFree(dy); free(A);free(x);free(yg);
}
static void mpfr_matmul(int N){
  size_t MA=(size_t)N*N*sizeof(double);
  double*A=(double*)malloc(MA),*B=(double*)malloc(MA),*Cg=(double*)malloc(MA);
  srand(22); for(long e=0;e<(long)N*N;++e){A[e]=frand();B[e]=frand();}
  double*dA,*dB,*dC; cudaMalloc(&dA,MA);cudaMalloc(&dB,MA);cudaMalloc(&dC,MA);
  cudaMemcpy(dA,A,MA,cudaMemcpyHostToDevice);cudaMemcpy(dB,B,MA,cudaMemcpyHostToDevice);
  arena_setup(LB,LT);
  km_matmul<<<LB,LT>>>(N,dA,dB,dC); cudaDeviceSynchronize();
  double tg=tmin([&]{km_matmul<<<LB,LT>>>(N,dA,dB,dC);});
  cudaMemcpy(Cg,dC,MA,cudaMemcpyDeviceToHost);
  double mx=0; int lim=N<=128?N:128; for(int i=0;i<lim;++i)for(int j=0;j<lim;++j){double ref=0; for(int k=0;k<N;++k)ref+=A[(size_t)i*N+k]*B[(size_t)k*N+j]; double e=ref!=0?fabs((Cg[(size_t)i*N+j]-ref)/ref):fabs(Cg[(size_t)i*N+j]); if(e>mx)mx=e;}
  emit("matmul","mpfr","mpfr",PREC,N,tg,mx,2.0*(double)N*(double)N*(double)N);
  cudaFree(dA);cudaFree(dB);cudaFree(dC); free(A);free(B);free(Cg);
}

/*==================== native double/float benches ====================*/
template<class T> static void nat_axpy(const char*name,int bits,int N){
  size_t MV=(size_t)N*sizeof(T);
  T*a=(T*)malloc(MV),*x=(T*)malloc(MV),*yg=(T*)malloc(MV);
  srand(33); for(int i=0;i<N;++i){a[i]=(T)frand();x[i]=(T)frand();}
  T al=(T)1.5;
  T*da,*dx,*dy; cudaMalloc(&da,MV);cudaMalloc(&dx,MV);cudaMalloc(&dy,MV);
  cudaMemcpy(da,a,MV,cudaMemcpyHostToDevice);cudaMemcpy(dx,x,MV,cudaMemcpyHostToDevice);
  int lb=(N+LT-1)/LT; if(lb<1)lb=1;
  kn_axpy<T><<<lb,LT>>>(N,da,al,dx,dy); cudaDeviceSynchronize();
  double tg=tmin([&]{kn_axpy<T><<<lb,LT>>>(N,da,al,dx,dy);});
  emit("axpy","native",name,bits,N,tg,0.0,2.0*(double)N);
  cudaFree(da);cudaFree(dx);cudaFree(dy); free(a);free(x);free(yg);
}
template<class T> static void nat_matvec(const char*name,int bits,int N){
  size_t MA=(size_t)N*N*sizeof(T), MV=(size_t)N*sizeof(T);
  T*A=(T*)malloc(MA),*x=(T*)malloc(MV),*yg=(T*)malloc(MV);
  srand(11); for(int i=0;i<N;++i){x[i]=(T)frand();for(int j=0;j<N;++j)A[(size_t)i*N+j]=(T)frand();}
  T*dA,*dx,*dy; cudaMalloc(&dA,MA);cudaMalloc(&dx,MV);cudaMalloc(&dy,MV);
  cudaMemcpy(dA,A,MA,cudaMemcpyHostToDevice);cudaMemcpy(dx,x,MV,cudaMemcpyHostToDevice);
  kn_matvec<T><<<LB,LT>>>(N,dA,dx,dy); cudaDeviceSynchronize();
  double tg=tmin([&]{kn_matvec<T><<<LB,LT>>>(N,dA,dx,dy);});
  emit("matvec","native",name,bits,N,tg,0.0,2.0*(double)N*(double)N);
  cudaFree(dA);cudaFree(dx);cudaFree(dy); free(A);free(x);free(yg);
}
template<class T> static void nat_matmul(const char*name,int bits,int N){
  size_t MA=(size_t)N*N*sizeof(T);
  T*A=(T*)malloc(MA),*B=(T*)malloc(MA),*Cg=(T*)malloc(MA);
  srand(22); for(long e=0;e<(long)N*N;++e){A[e]=(T)frand();B[e]=(T)frand();}
  T*dA,*dB,*dC; cudaMalloc(&dA,MA);cudaMalloc(&dB,MA);cudaMalloc(&dC,MA);
  cudaMemcpy(dA,A,MA,cudaMemcpyHostToDevice);cudaMemcpy(dB,B,MA,cudaMemcpyHostToDevice);
  kn_matmul<T><<<LB,LT>>>(N,dA,dB,dC); cudaDeviceSynchronize();
  double tg=tmin([&]{kn_matmul<T><<<LB,LT>>>(N,dA,dB,dC);});
  emit("matmul","native",name,bits,N,tg,0.0,2.0*(double)N*(double)N*(double)N);
  cudaFree(dA);cudaFree(dB);cudaFree(dC); free(A);free(B);free(Cg);
}

int main(int argc,char**argv){
  std::vector<int> ax={4096,16384,65536}, mv={128,256,512}, mm={64,128,256};
  const char*out=nullptr; bool append=false,do_native=false; int dev=0;
  for(int i=1;i<argc;i++){
    if(!strcmp(argv[i],"--reps")&&i+1<argc)g_reps=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--lb")&&i+1<argc)LB=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--lt")&&i+1<argc)LT=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--dev")&&i+1<argc)dev=atoi(argv[++i]);
    else if(!strcmp(argv[i],"--out")&&i+1<argc)out=argv[++i];
    else if(!strcmp(argv[i],"--append"))append=true;
    else if(!strcmp(argv[i],"--native"))do_native=true;
  }
  cudaSetDevice(dev);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,(size_t)128*1024*1024);
  cudaDeviceSetLimit(cudaLimitStackSize,(size_t)200*1024);
  g_out = out? fopen(out,append?"a":"w") : stdout;
  if(!append && out) fprintf(g_out,"op,kind,type,bits,N,gpu_sec,gpu_mflops,relerr\n");
  fprintf(stderr,"# mpfr_gpu_bench PREC=%d reps=%d grid=%dx%d native=%d\n",PREC,g_reps,LB,LT,do_native);
  if(do_native){
    for(int N:ax){ nat_axpy<double>("double",53,N); nat_axpy<float>("float",24,N); }
    for(int N:mv){ nat_matvec<double>("double",53,N); nat_matvec<float>("float",24,N); }
    for(int N:mm){ nat_matmul<double>("double",53,N); nat_matmul<float>("float",24,N); }
    fprintf(stderr,"  native done\n");
  }
  for(int N:ax){ mpfr_axpy(N);   fprintf(stderr,"  mpfr axpy N=%d done\n",N); }
  for(int N:mv){ mpfr_matvec(N); fprintf(stderr,"  mpfr matvec N=%d done\n",N); }
  for(int N:mm){ mpfr_matmul(N); fprintf(stderr,"  mpfr matmul N=%d done\n",N); }
  if(out)fclose(g_out);
  return 0;
}

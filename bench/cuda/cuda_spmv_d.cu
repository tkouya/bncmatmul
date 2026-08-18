/*****************************************************************
 * cuda_spmv_d.cu -- GPU CSR SpMV vs CPU OpenMP DRS SpMV, native double.
 *   y = A*x , synthetic sparse matrix (tridiagonal + one scattered entry/row).
 *   GPU : mul_gdspmatrix_gdvec           CPU : _bncomp_mul_drsmatrix_dvec (OpenMP)
 * CSV: RESULT,spmv,d,dim,nnz,cpu_time,gpu_time,omp,relerr
 *****************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>

extern "C" {
#include "bncsparse.h"
}
#include "gdlinear.h"
#include "gdsparse.h"
#include <omp.h>

extern "C" void _bncomp_mul_drsmatrix_dvec(DVector, DRSMatrix, DVector);

static FILE *g_csv=nullptr;
using clk=std::chrono::high_resolution_clock;
static double secs(clk::time_point t0){return std::chrono::duration<double>(clk::now()-t0).count();}

static void drs_to_csr(DRSMatrix m,long int**rp_,long int**ci_,double**vv_,long int*nnz_){
    long int i,j,k=0,base=0,nnz=m->nzero_total_num;
    long int*rp=(long int*)malloc((m->row_dim+1)*sizeof(long int));
    long int*ci=(long int*)malloc(nnz*sizeof(long int));
    double*vv=(double*)malloc(nnz*sizeof(double));
    rp[0]=0;
    for(i=0;i<m->row_dim;i++){ for(j=0;j<m->nzero_col_dim[i];j++){ci[k]=m->nzero_index[i][j];vv[k]=m->element[base+j];k++;} base+=m->real_nzero_col_dim[i]; rp[i+1]=k; }
    *rp_=rp;*ci_=ci;*vv_=vv;*nnz_=nnz;
}
static double relerr(DVector ref,GDVector gpu){
    long n=ref->dim; GDVector h=init_gdvector(n); double mx=0;
    cudaMemcpy((void*)h->element,(void*)gpu->element,sizeof(double)*n,cudaMemcpyDeviceToHost);
    for(long i=0;i<n;i++){double a=ref->element[i],b=h->element[i],e=fabs(a-b); if(a!=0)e/=fabs(a); if(e>mx)mx=e;}
    free_gdvector(h); return mx;
}

int main(int argc,char**argv){
    std::vector<long> sizes={4096,16384,65536,262144,1048576};
    int reps=20,nbg=64,ntb=128;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--reps")&&i+1<argc)reps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--blocks")&&i+1<argc)nbg=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--threads")&&i+1<argc)ntb=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--sizes")&&i+1<argc){sizes.clear();const char*p=argv[++i];while(*p){char*e=0;long v=strtol(p,&e,10);if(e==p||v<=0)break;sizes.push_back(v);p=e;while(*p==','||*p==' ')++p;}}
        else if(!strcmp(argv[i],"--max-cpu")&&i+1<argc)atol(argv[++i]); /* accepted/ignored: SpMV CPU is cheap */
    }
    const char*env=getenv("OMP_NUM_THREADS");int nt=env?atoi(env):omp_get_max_threads();if(nt<1)nt=omp_get_max_threads();omp_set_num_threads(nt);
    cudaSetDevice(0); cudaDeviceProp p; cudaGetDeviceProperties(&p,0); g_csv=stdout;
    printf(" native double SpMV : device %s sm_%d%d  CPU OpenMP threads=%d\n",p.name,p.major,p.minor,nt);

    for(long dim:sizes){
        DMatrix dense=init_dmatrix(dim,dim); DVector x=init_dvector(dim),y=init_dvector(dim);
        for(long i=0;i<dim;i++){ set_dmatrix_ij(dense,i,i,2.0); if(i>0)set_dmatrix_ij(dense,i,i-1,-1.0); if(i<dim-1)set_dmatrix_ij(dense,i,i+1,-1.0);
            set_dmatrix_ij(dense,i,(i*7+3)%dim,0.5); set_dvector_i(x,i,sqrt((double)(i+1))); }
        DRSMatrix drs=init_set_drsmatrix_dmatrix(dense);
        long int *row_ptr,*col_idx,nnz; double*val; drs_to_csr(drs,&row_ptr,&col_idx,&val,&nnz);
        GDSPMatrix gA=init_gdspmatrix_dev(dim,dim,nnz,row_ptr,col_idx,val);
        GDVector gx=init_gdvector_dev(dim),gy=init_gdvector_dev(dim); subst_gdvector_dev_dvec(gx,x);

        _bncomp_mul_drsmatrix_dvec(y,drs,x);
        mul_gdspmatrix_gdvec(gy,gA,gx,nbg,ntb); cudaDeviceSynchronize();
        double re=relerr(y,gy);

        double tg=1e30; for(int r=0;r<reps;r++){auto t=clk::now();mul_gdspmatrix_gdvec(gy,gA,gx,nbg,ntb);cudaDeviceSynchronize();double s=secs(t);if(s<tg)tg=s;}
        double tc=1e30; for(int r=0;r<reps;r++){auto t=clk::now();_bncomp_mul_drsmatrix_dvec(y,drs,x);double s=secs(t);if(s<tc)tc=s;}
        fprintf(g_csv,"RESULT,spmv,d,%ld,%ld,%.9g,%.9g,omp,%.6e\n",dim,nnz,tc,tg,re); fflush(g_csv);

        free(row_ptr);free(col_idx);free(val);
        free_dmatrix(dense);free_dvector(x);free_dvector(y);
        free_gdspmatrix_dev(gA);free_gdvector_dev(gx);free_gdvector_dev(gy);
    }
    return 0;
}

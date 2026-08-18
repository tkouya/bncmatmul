/* gcflu.cu -- GPU native complex float LU (partial pivoting) + triangular solve.
 * Complex stored as separate re/im float arrays (GCFMatrix/GCFVector).
 * Right-looking elimination; pivot by |z|^2 = re^2+im^2; complex multiplier/division.
 * (add -DGCFLU_TEST for the self-test main)
 */
#include <cstdio>
#include <cmath>
#ifndef LU_THRESH
#define LU_THRESH 1e-3
#endif
#include "gcflinear.h"

// complex helpers (re/im doubles)
__device__ static inline void cdiv(float ar, float ai, float br, float bi, float *qr, float *qi)
{
	float d = br * br + bi * bi;
	*qr = (ar * br + ai * bi) / d;
	*qi = (ai * br - ar * bi) / d;
}

__global__ static void gcflu_find_pivot(const float *Are, const float *Aim, int n, int k, int *imax)
{
	if(blockIdx.x * blockDim.x + threadIdx.x != 0) return;
	int p = k;
	float best = Are[(size_t)k * n + k] * Are[(size_t)k * n + k] + Aim[(size_t)k * n + k] * Aim[(size_t)k * n + k];
	for(int i = k + 1; i < n; ++i)
	{
		float v = Are[(size_t)i * n + k] * Are[(size_t)i * n + k] + Aim[(size_t)i * n + k] * Aim[(size_t)i * n + k];
		if(v > best) { best = v; p = i; }
	}
	*imax = p;
}

__global__ static void gcflu_swap_row(float *Are, float *Aim, float *bre, float *bim, int n, int k, int m)
{
	int stride = gridDim.x * blockDim.x;
	for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride)
	{
		float t;
		t = Are[(size_t)k * n + j]; Are[(size_t)k * n + j] = Are[(size_t)m * n + j]; Are[(size_t)m * n + j] = t;
		t = Aim[(size_t)k * n + j]; Aim[(size_t)k * n + j] = Aim[(size_t)m * n + j]; Aim[(size_t)m * n + j] = t;
	}
	if(blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		float t;
		t = bre[k]; bre[k] = bre[m]; bre[m] = t;
		t = bim[k]; bim[k] = bim[m]; bim[m] = t;
	}
}

__global__ static void gcflu_eliminate(float *Are, float *Aim, int n, int k)
{
	int stride = gridDim.x * blockDim.x;
	for(int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
	{
		float mr, mi;
		cdiv(Are[(size_t)i * n + k], Aim[(size_t)i * n + k], Are[(size_t)k * n + k], Aim[(size_t)k * n + k], &mr, &mi);
		Are[(size_t)i * n + k] = mr; Aim[(size_t)i * n + k] = mi;
		for(int j = k + 1; j < n; ++j)
		{
			float ar = Are[(size_t)k * n + j], ai = Aim[(size_t)k * n + j];
			Are[(size_t)i * n + j] -= mr * ar - mi * ai;
			Aim[(size_t)i * n + j] -= mr * ai + mi * ar;
		}
	}
}

__global__ static void gcflu_fbsub(const float *Are, const float *Aim, const float *bre, const float *bim,
                                   float *xre, float *xim, int n)
{
	if(blockIdx.x * blockDim.x + threadIdx.x != 0) return;
	// forward (unit lower L)
	for(int i = 0; i < n; ++i)
	{
		float sr = bre[i], si = bim[i];
		for(int j = 0; j < i; ++j)
		{
			float ar = Are[(size_t)i * n + j], ai = Aim[(size_t)i * n + j];
			sr -= ar * xre[j] - ai * xim[j];
			si -= ar * xim[j] + ai * xre[j];
		}
		xre[i] = sr; xim[i] = si;
	}
	// back (upper U)
	for(int i = n - 1; i >= 0; --i)
	{
		float sr = xre[i], si = xim[i];
		for(int j = i + 1; j < n; ++j)
		{
			float ar = Are[(size_t)i * n + j], ai = Aim[(size_t)i * n + j];
			sr -= ar * xre[j] - ai * xim[j];
			si -= ar * xim[j] + ai * xre[j];
		}
		float qr, qi;
		cdiv(sr, si, Are[(size_t)i * n + i], Aim[(size_t)i * n + i], &qr, &qi);
		xre[i] = qr; xim[i] = qi;
	}
}

extern "C" int
gcf_lu_solve_dev(GCFMatrix A_dev, GCFVector b_dev, GCFVector x_dev, long int *ch, int blocks, int threads)
{
	int n = (int)A_dev->col_dim;
	if(blocks <= 0) blocks = 64;
	if(threads <= 0) threads = 128;
	int *d_imax, h_imax;
	cudaMalloc(&d_imax, sizeof(int));
	float *Are = A_dev->re, *Aim = A_dev->im, *bre = b_dev->re, *bim = b_dev->im, *xre = x_dev->re, *xim = x_dev->im;
	for(int k = 0; k < n; ++k)
	{
		gcflu_find_pivot<<<1, 1>>>(Are, Aim, n, k, d_imax);
		cudaMemcpy(&h_imax, d_imax, sizeof(int), cudaMemcpyDeviceToHost);
		if(ch) ch[k] = h_imax;
		if(h_imax != k) gcflu_swap_row<<<blocks, threads>>>(Are, Aim, bre, bim, n, k, h_imax);
		gcflu_eliminate<<<blocks, threads>>>(Are, Aim, n, k);
	}
	gcflu_fbsub<<<1, 1>>>(Are, Aim, bre, bim, xre, xim, n);
	cudaError_t err = cudaDeviceSynchronize();
	cudaFree(d_imax);
	return err == cudaSuccess ? 0 : (int)err;
}

#ifdef GCFLU_TEST
#include "cdlinear.h"
extern "C" void mul_cdmatrix_cdvec(CDVector, CDMatrix, CDVector);
static inline double _Complex _mkc(float re, float im) { double _Complex z; __real__ z = re; __imag__ z = im; return z; }
static inline float _cabs(double _Complex z) { float r = __real__ z, i = __imag__ z; return sqrt(r * r + i * i); }
int main(void)
{
	long int i, j, dim = 100;
	CDMatrix a = init_cdmatrix(dim, dim);
	CDVector xtrue = init_cdvector(dim), b = init_cdvector(dim);
	// diagonally dominant complex A, known complex solution
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
			set_cdmatrix_ij(a, i, j, (i == j) ? _mkc(2.0 * dim, 1.0) : _mkc(1.0 / (float)(i + j + 2), -1.0 / (float)(i + j + 3)));
	for(i = 0; i < dim; i++) set_cdvector_i(xtrue, i, _mkc((float)i, (float)(dim - i)));
	mul_cdmatrix_cdvec(b, a, xtrue);

	GCFMatrix ga = init_gcfmatrix_dev(dim, dim);
	GCFVector gb = init_gcfvector_dev(dim), gx = init_gcfvector_dev(dim);
	subst_gcfmatrix_dev_cdmat(ga, a);
	subst_gcfvector_dev_cdvec(gb, b);
	if(gcf_lu_solve_dev(ga, gb, gx, NULL, 64, 128)) { fprintf(stderr, "GPU complex LU failed\n"); return 1; }
	CDVector xg = init_cdvector(dim);
	subst_cdvector_gcfvec_dev(xg, gx);

	float mr = 0.0;
	for(i = 0; i < dim; i++)
	{
		double _Complex g = get_cdvector_i(xg, i), c = get_cdvector_i(xtrue, i);
		float d = _cabs(g - c);
		if(_cabs(c) != 0.0) d /= _cabs(c);
		if(d > mr) mr = d;
	}
	printf("GPU gcf complex LU vs known x (dim=%ld): max rel err = %10.3e -> %s\n",
		dim, mr, mr < LU_THRESH ? "PASS (native complex float)" : "FAIL");
	return mr < LU_THRESH ? 0 : 1;
}
#endif

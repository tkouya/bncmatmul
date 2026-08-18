# env.sh -- portable configuration for the GPU(gdtq/mpc_cuda) vs CPU-OMP benchmark.
# Source this from the other build/run scripts. Override any variable by exporting
# it BEFORE calling the scripts, e.g.:  PREFIX=/opt/local ARCH=sm_80 ./build_gpu_omp.sh
#
# PREFIX    : install prefix that contains  include/gdtq/ , include/mpc_cuda* ,
#             include/rds.h , include/qd/ , lib/{libqd,libmpc,libmpfr,libgmp,libmpc_cuda}.*
# CUDA_HOME : CUDA toolkit root (has bin/nvcc, lib64/libcudart)
# ARCH      : GPU compute capability. H100=sm_90. CHANGE THIS for your GPU
#             (A100=sm_80, RTX40=sm_89, GH200/GB=sm_90a/sm_100, etc.)
# OMPLIB    : OpenMP EFT static lib built from this repo (make avx512 + omp target)
# SERLIB    : serial EFT static lib (provides host init_/set_/subst_ helpers)
: "${PREFIX:=/home/tkouya/local}"
: "${CUDA_HOME:=/usr/local/cuda}"
: "${ARCH:=sm_90}"
: "${OMPLIB:=./libbncmatmul-0.24-omp_avx512.a}"
: "${SERLIB:=./libbncmatmul-0.24_avx512.a}"
: "${OMP_NUM_THREADS:=$(nproc)}"
export PREFIX CUDA_HOME ARCH OMPLIB SERLIB OMP_NUM_THREADS
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# common compile fragments
NVCC="nvcc -O3 -arch=$ARCH -std=c++17 -fmad=false -D__NV_NO_VECTOR_DEPRECATION_DIAG"
INC="-Iinclude -I$PREFIX/include/gdtq -I$PREFIX/include -I$CUDA_HOME/include"
DEFS="-DUSE_DDLINEAR -DUSE_TDLINEAR -DUSE_QDLINEAR -DUSE_DSLINEAR -DUSE_TSLINEAR -DUSE_QSLINEAR -DUSE_GMP -DUSE_MPFR"
SUP="-diag-suppress 20011 -diag-suppress 177 -diag-suppress 550 -diag-suppress 20208 -diag-suppress 20044 -Wno-deprecated-declarations"
CPULK="-Xlinker --start-group $OMPLIB $SERLIB -Xlinker --end-group -Xcompiler -fopenmp -lgomp"
SYSLK="-L$CUDA_HOME/lib64 -lcudart -lcudadevrt -L$PREFIX/lib -lmpc -lmpfr -lgmp -lqd -lpthread -lstdc++ -lm"
MPCLIB="$PREFIX/lib/libmpc_cuda.a"

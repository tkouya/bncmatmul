dnl SPDX-License-Identifier: BSD-3-Clause
dnl AX_CUDA --- detect NVIDIA CUDA toolkit and set NVCC, NVCC_FLAGS,
dnl             CUDA_DIR, CUDA_CPPFLAGS, CUDA_LDFLAGS, CUDA_LIBS,
dnl             CUDA_ARCH and CUDA_SDK_HOME for the rest of the build.
dnl
dnl Copied verbatim from gdtq-0.0.2/m4/ax_cuda.m4 so that bncmatmul and
dnl gdtq can be configured with the same options.
dnl
dnl Options provided to ./configure:
dnl   --with-cuda=PATH         CUDA toolkit install prefix (default: /usr/local/cuda)
dnl   --with-cuda-arch=ARCH    target compute architecture (default: sm_80)
dnl   --with-nvcc=PROG         path to nvcc (default: $CUDA_DIR/bin/nvcc)
dnl   --with-cuda-sdk=PATH     legacy CUDA Samples common/ path (optional)
dnl   --with-nvcc-flags=FLAGS  extra flags to pass to nvcc

AC_DEFUN([AX_CUDA], [
  AC_ARG_WITH([cuda],
    [AS_HELP_STRING([--with-cuda=PATH],
      [CUDA toolkit install prefix @<:@default=/usr/local/cuda@:>@])],
    [CUDA_DIR="$withval"], [CUDA_DIR="/usr/local/cuda"])

  AC_ARG_WITH([cuda-arch],
    [AS_HELP_STRING([--with-cuda-arch=ARCH],
      [target compute architecture, e.g. sm_80, sm_90, sm_121 @<:@default=sm_80@:>@])],
    [CUDA_ARCH="$withval"], [CUDA_ARCH="sm_80"])

  AC_ARG_WITH([nvcc],
    [AS_HELP_STRING([--with-nvcc=PROG], [path to nvcc])],
    [NVCC="$withval"], [NVCC=""])

  AC_ARG_WITH([cuda-sdk],
    [AS_HELP_STRING([--with-cuda-sdk=PATH],
      [legacy CUDA Samples common/ directory (optional)])],
    [CUDA_SDK_HOME="$withval"], [CUDA_SDK_HOME=""])

  AC_ARG_WITH([nvcc-flags],
    [AS_HELP_STRING([--with-nvcc-flags=FLAGS], [extra flags appended to NVCC_FLAGS])],
    [EXTRA_NVCC_FLAGS="$withval"], [EXTRA_NVCC_FLAGS=""])

  AS_IF([test -z "$NVCC"], [NVCC="$CUDA_DIR/bin/nvcc"])
  AC_MSG_CHECKING([for nvcc])
  AS_IF([test -x "$NVCC"], [
    AC_MSG_RESULT([$NVCC])
  ], [
    AC_PATH_PROG([NVCC_PATH], [nvcc], [no])
    AS_IF([test "x$NVCC_PATH" != "xno"], [
      NVCC="$NVCC_PATH"
      AC_MSG_RESULT([$NVCC (from PATH)])
    ], [
      AC_MSG_RESULT([no])
      AC_MSG_ERROR([nvcc not found; pass --with-cuda=PATH or --with-nvcc=PROG])
    ])
  ])

  AC_MSG_CHECKING([CUDA include directory])
  AS_IF([test -f "$CUDA_DIR/include/cuda.h"], [
    AC_MSG_RESULT([$CUDA_DIR/include])
  ], [
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([cuda.h not found under $CUDA_DIR/include])
  ])

  AC_MSG_CHECKING([CUDA library directory])
  CUDA_LIBDIR=""
  for d in "$CUDA_DIR/lib64" "$CUDA_DIR/lib/x64" "$CUDA_DIR/lib"; do
    if test -d "$d"; then
      CUDA_LIBDIR="$d"
      break
    fi
  done
  AS_IF([test -n "$CUDA_LIBDIR"], [
    AC_MSG_RESULT([$CUDA_LIBDIR])
  ], [
    AC_MSG_RESULT([not found])
    AC_MSG_ERROR([CUDA library directory not found under $CUDA_DIR])
  ])

  CUDA_CPPFLAGS="-I$CUDA_DIR/include"
  CUDA_LDFLAGS="-L$CUDA_LIBDIR"
  CUDA_LIBS="-lcuda -lcudart -lcublas"
  NVCC_FLAGS="-fmad=false -D__NV_NO_VECTOR_DEPRECATION_DIAG -O3 $EXTRA_NVCC_FLAGS"

  AS_IF([test -n "$CUDA_SDK_HOME"], [
    AS_IF([test -d "$CUDA_SDK_HOME/inc"], [
      CUDA_SDK_INCLUDE="-I$CUDA_SDK_HOME/inc"
    ], [
      CUDA_SDK_INCLUDE="-I$CUDA_SDK_HOME"
    ])
    AS_IF([test -d "$CUDA_SDK_HOME/lib"], [
      CUDA_SDK_LDFLAGS="-L$CUDA_SDK_HOME/lib"
    ], [
      CUDA_SDK_LDFLAGS=""
    ])
  ], [
    CUDA_SDK_INCLUDE=""
    CUDA_SDK_LDFLAGS=""
  ])

  AC_SUBST([NVCC])
  AC_SUBST([NVCC_FLAGS])
  AC_SUBST([CUDA_DIR])
  AC_SUBST([CUDA_LIBDIR])
  AC_SUBST([CUDA_ARCH])
  AC_SUBST([CUDA_CPPFLAGS])
  AC_SUBST([CUDA_LDFLAGS])
  AC_SUBST([CUDA_LIBS])
  AC_SUBST([CUDA_SDK_HOME])
  AC_SUBST([CUDA_SDK_INCLUDE])
  AC_SUBST([CUDA_SDK_LDFLAGS])
])

/***************************************************************************
Copyright (c) 2015,                               The OpenBLAS Project
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/


struct gpu_context
{
        int                     platform;
        int                     device;
        int                     unified_mem;
        cl_platform_id          platform_id;
        cl_device_id            device_id;
        cl_context              context;
        cl_program              program;
        cl_kernel               kernel;
        cl_command_queue        command_queue;
        cl_mem                  A;
        cl_mem                  B;
        cl_mem                  C;
        void*                   hA;
        void*                   hB;
        void*                   hC;
        char                    device_name[64];
};

static int  create_gpu_context(struct gpu_context *gpu);
static int  create_gpu_program_nonunified(struct gpu_context *gpu, char *func ,size_t bufsize);
static void destroy_gpu_context(struct gpu_context *gpu);
static void release_gpu_program(struct gpu_context *gpu);
static void open_gpu()  __attribute__((constructor));
static void close_gpu() __attribute__((destructor));

static void sgemm_gpu_ccopy(int M, int N, float *A, int LDA, float *B, blasint LDB, float beta) __attribute__ ((noinline));
static void sgemm_gpu_acopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void sgemm_gpu_btcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void sgemm_gpu_bcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static int  sgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, float ALPHA, int acopy, int bcopy, double *ktime) __attribute__ ((noinline));

static void dgemm_gpu_ccopy(int M, int N, double *A, int LDA, double *B, blasint LDB, double beta) __attribute__ ((noinline));
static void dgemm_gpu_acopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void dgemm_gpu_btcopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void dgemm_gpu_bcopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static int  dgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, double ALPHA, int acopy, int bcopy, double *ktime) __attribute__ ((noinline));

#if defined(CGEMM_GLOBAL0_DIV)

static void cgemm_gpu_ccopy(int M, int N, float *A, int LDA, float *B, blasint LDB, float *beta) __attribute__ ((noinline));
static void cgemm_gpu_acopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void cgemm_gpu_btcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void cgemm_gpu_btcopy_conj(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void cgemm_gpu_bcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void cgemm_gpu_bcopy_conj(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static int  cgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, float *ALPHA, int acopy, int bcopy, double *ktime) __attribute__ ((noinline));

#endif

#if defined(ZGEMM_GLOBAL0_DIV)

static void zgemm_gpu_ccopy(int M, int N, double *A, int LDA, double *B, blasint LDB, double *beta) __attribute__ ((noinline));
static void zgemm_gpu_acopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void zgemm_gpu_btcopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void zgemm_gpu_btcopy_conj(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void zgemm_gpu_bcopy(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void zgemm_gpu_bcopy_conj(int M, int N, double *A , blasint LDA, double *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static int  zgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, double *ALPHA, int acopy, int bcopy, double *ktime) __attribute__ ((noinline));


#endif


#if defined(HAVE_DSGEMM)

static void dsgemm_gpu_ccopy(int M, int N, float *A, int LDA, double *B, blasint LDB, double alpha, double beta) __attribute__ ((noinline));
static void dsgemm_gpu_acopy(int M, int N, double *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void dsgemm_gpu_btcopy(int M, int N, double *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));
static void dsgemm_gpu_bcopy(int M, int N, double *A , blasint LDA, float *B, int PAD_M, int PAD_N) __attribute__ ((noinline));

#endif

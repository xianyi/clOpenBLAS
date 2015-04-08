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

#define ALLOC_SIZE 4096*4096*4

#define HAVE_DSGEMM 1


#define SGEMM_M_MAX 4096
#define SGEMM_N_MAX 4096
#define SGEMM_K_MAX 4096

#define DGEMM_M_MAX 2048
#define DGEMM_N_MAX 2048
#define DGEMM_K_MAX 4096

#define CGEMM_M_MAX 2048
#define CGEMM_N_MAX 2048
#define CGEMM_K_MAX 4096

#define ZGEMM_M_MAX 1024
#define ZGEMM_N_MAX 1024
#define ZGEMM_K_MAX 4096

#define SGEMM_N_BUFFERS 16
#define DGEMM_N_BUFFERS 16
#define CGEMM_N_BUFFERS 16
#define ZGEMM_N_BUFFERS 16

#define GALLOC_SIZE_A ( SGEMM_M_MAX * SGEMM_K_MAX * sizeof(float) )
#define GALLOC_SIZE_B ( SGEMM_N_MAX * SGEMM_K_MAX * sizeof(float) )
#define GALLOC_SIZE_C ( SGEMM_M_MAX * SGEMM_N_MAX * sizeof(float) )

#if defined(HAVE_DSGEMM)

#define MALLOC_SIZE_A ( GALLOC_SIZE_A * 2 )
#define MALLOC_SIZE_B ( GALLOC_SIZE_B * ( SGEMM_N_BUFFERS + 1 ) *2 )
#define MALLOC_SIZE_C ( GALLOC_SIZE_C *2 )

#else

#define MALLOC_SIZE_A ( GALLOC_SIZE_A * 1 )
#define MALLOC_SIZE_B ( GALLOC_SIZE_B * ( SGEMM_N_BUFFERS + 1 ) *1 )
#define MALLOC_SIZE_C ( GALLOC_SIZE_C *1 )

#endif

#define SGEMM_GLOBAL0_DIV 8
#define SGEMM_GLOBAL1_DIV 8

#define SGEMM_LOCAL0 8
#define SGEMM_LOCAL1 8

#define SGEMM_PAD_M 64
#define SGEMM_PAD_N 64
#define SGEMM_PAD_K 4

#define DGEMM_GLOBAL0_DIV 4
#define DGEMM_GLOBAL1_DIV 4

#define DGEMM_LOCAL0 8
#define DGEMM_LOCAL1 8

#define DGEMM_PAD_M 32
#define DGEMM_PAD_N 32
#define DGEMM_PAD_K 4 

#define CGEMM_GLOBAL0_DIV 4
#define CGEMM_GLOBAL1_DIV 4

#define CGEMM_LOCAL0 8
#define CGEMM_LOCAL1 8

#define CGEMM_PAD_M 32
#define CGEMM_PAD_N 32
#define CGEMM_PAD_K 4 

#define ZGEMM_GLOBAL0_DIV 2
#define ZGEMM_GLOBAL1_DIV 2

#define ZGEMM_LOCAL0 8
#define ZGEMM_LOCAL1 8

#define ZGEMM_PAD_M 16
#define ZGEMM_PAD_N 16
#define ZGEMM_PAD_K 4 


static char  *DEFAULT_KERNEL = "oland";

static char  *DEFAULT_DEVICE = "oland";

#include "../include/gpu_common.h"




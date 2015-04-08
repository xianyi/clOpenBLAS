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

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <clopenblas_common.h>
#include "geforce_4xx.h"

// #define PROFILE
// #define DEBUG

static struct gpu_context gpu;
static int have_gpu_context = 0;
static cl_context gpu_context;

#include "../gpu_common/gemm_common.c"

#include "../gpu_common/sgemm.c"
#include "../gpu_common/dgemm.c"
#include "../gpu_common/dsgemm.c"

void * blas_gpu_info(int level, char *bfunc, blasint *M, blasint *N, blasint *K)
{

	if ( have_gpu_context == 0)
		return(NULL);

	if ( level != 3 )
		return(NULL);

	int (*foo)();

	switch ( bfunc[0] )
	{

		case 's':
		case 'S':
			if ( !strncasecmp(bfunc,"sgemm", 6))
			{
				if ( (M != NULL ) && (*M < SGEMM_PAD_M) ) return(NULL);
				if ( (N != NULL ) && (*N < SGEMM_PAD_N) ) return(NULL);
				if ( (K != NULL ) && (*K < SGEMM_PAD_K) ) return(NULL);

				foo = &sgemm_gpu_simple;
				return(foo);
			}
			break;

                case 'd':
                case 'D':
                        if ( !strncasecmp(bfunc,"dgemm", 6))
                        {
                                if ( (M != NULL ) && (*M < DGEMM_PAD_M) ) return(NULL);
                                if ( (N != NULL ) && (*N < DGEMM_PAD_N) ) return(NULL);
                                if ( (K != NULL ) && (*K < DGEMM_PAD_K) ) return(NULL);

                                foo = &dgemm_gpu_simple;
                                return(foo);
                        }
                        if ( !strncasecmp(bfunc,"dsgemm", 7))
                        {
                                if ( (M != NULL ) && (*M < SGEMM_PAD_M) ) return(NULL);
                                if ( (N != NULL ) && (*N < SGEMM_PAD_N) ) return(NULL);
                                if ( (K != NULL ) && (*K < SGEMM_PAD_K) ) return(NULL);

                                foo = &dsgemm_gpu_simple;
                                return(foo);
                        }

                        break;


		default: return(NULL);
	}
	return(NULL);
}




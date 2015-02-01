/***************************************************************************
Copyright (c) 2013, The OpenBLAS Project
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


#define _GNU_SOURCE 1
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>
#include <ctype.h>
#include <dlfcn.h>

#include <common.h>

static void * handle = NULL;
static void * ohandle = NULL;

static void * (*blas_gpu_info)(int level3, char *st, blasint *m, blasint *n, blasint *k);

void sgemm_(char * transa, char * transb, blasint *m, blasint *n, blasint *k, float *alpha, float *a, blasint *lda, float *b, blasint *ldb, float *beta, float * c, blasint *ldc );
void dgemm_(char * transa, char * transb, blasint *m, blasint *n, blasint *k, double *alpha, double *a, blasint *lda, double *b, blasint *ldb, double *beta, double * c, blasint *ldc );

static void open_wrap()  __attribute__((constructor));
static void close_wrap() __attribute__((destructor));

static void open_wrap()
{

	ohandle = dlopen( "libopenblas.so", RTLD_LAZY);


	char *p;
	p=getenv("OPENBLAS_GPU_LIB");
	if ( p == NULL)
	{
		#ifdef DEBUG
			fprintf(stderr, "Variable OPENBLAS_GPU_LIB not found\n");
		#endif
		handle = NULL;
		return;
		
	}
	handle = dlopen( p, RTLD_LAZY);		
	if ( handle == NULL )
	{
		#ifdef DEBUG
			fprintf(stderr,"%s\n",dlerror());
		#endif
		return;
	}
	blas_gpu_info = dlsym(handle, "blas_gpu_info");			
	if ( blas_gpu_info == NULL )
	{
		#ifdef DEBUG
			fprintf(stderr, "%s\n",dlerror());
		#endif

	}	

}

static void close_wrap()
{
	if ( handle != NULL )
		dlclose(handle);

}

#include "sgemm.c"
#include "dgemm.c"


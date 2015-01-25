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

static void * (*blas_gpu_info)(int level3, char *st, blasint *m, blasint *n, blasint *k);

// static int (*sgemm_gpu) (char * transa, char * transb, blasint *m, blasint *n, blasint *k, float *alpha, float *a, blasint *lda, float *b, blasint *ldb, float *beta, float * c, blasint *ldc );

// static void (*sgemm_cpu) ( char * transa, char * transb, blasint *m, blasint *n, blasint *k, float *alpha, float *a, blasint *lda, float *b, blasint *ldb, float *beta, float * c, blasint *ldc );

void sgemm_(char * transa, char * transb, blasint *m, blasint *n, blasint *k, float *alpha, float *a, blasint *lda, float *b, blasint *ldb, float *beta,float * c, blasint *ldc );

static void open_wrap()  __attribute__((constructor));
static void close_wrap() __attribute__((destructor));

static void open_wrap()
{

	char *p;
	p=getenv("OPENBLAS_GPU_LIB");
	if ( p == NULL)
	{
		#ifdef DEBUG
			printf("Variable OPENBLAS_GPU_LIB not found\n");
		#endif
		handle = NULL;
		return;
		
	}
	handle = dlopen( p, RTLD_LAZY);		
	if ( handle == NULL )
	{
			printf("%s\n",dlerror());
			return;
	}
	blas_gpu_info = dlsym(handle, "blas_gpu_info");			
	if ( blas_gpu_info == NULL )
	{
			printf("%s\n",dlerror());

	}	

}

static void close_wrap()
{
	if ( handle != NULL )
		dlclose(handle);

}

void sgemm_(char *TRANSA, char *TRANSB, blasint *M, blasint *N, blasint *K, float *alpha, float *a, blasint *LDA, float *b, blasint *LDB, float *beta, float *c, blasint *LDC )
{
	int ret;
	int (*sgemm_gpu)();
	void (*sgemm_cpu)();
	char *p;

        int info = 0;
        int transa = -1;
        int transb = -1;
        char transA = toupper(*TRANSA);
        char transB = toupper(*TRANSB);

	#ifdef DEBUG
		FILE *fp1;
		fp1 = fopen("/tmp/wrap.log", "a+");
		fprintf(fp1, "sgemm called M=%d N=%d K=%d\n", *M, *N, *K);
		printf("sgemm called M=%d N=%d K=%d\n", *M, *N, *K);
		if (fp1)
			fclose(fp1);
	#endif


        if ( transA == 'N' ) transa = 0;
        if ( transA == 'T' ) transa = 1;
        if ( transB == 'N' ) transb = 0;
        if ( transB == 'T' ) transb = 1;

        if ( transb < 0 ) info = 2;
        if ( transa < 0 ) info = 1;

        if ( info > 0 )
        {
                printf("** On entry to sgemm_ parameter number %d had a illegal value\n",info);
                return;
        }

        if ( *LDC < *M ) info = 13;
	if ((( transb & 1 ) && ( *LDB < *N )) || (( transb == 0 ) &&  ( *LDB < *K ))) info = 10;
        if ((( transa & 1 ) && ( *LDA < *K )) || (( transa == 0 ) &&  ( *LDA < *M ))) info = 8;
        if ( *K < 1 ) info = 5;
        if ( *N < 1 ) info = 4;
        if ( *M < 1 ) info = 3;

        if ( info > 0 )
        {
                printf("** On entry to sgemm_ parameter number %d had a illegal value\n",info);
                return;
        }


	#ifdef DEBUG
		FILE *fp;
		fp = fopen("/tmp/wrap.log", "a+");
		fprintf(fp, "Running sgemm M=%d N=%d K=%d\n", *M, *N, *K);
		printf("Running sgemm M=%d N=%d K=%d\n", *M, *N, *K);
	#endif


	blasint minvalue;
	int use_gpu = 1;

	p = getenv("SGEMM_GPU_MINSIZE");
	if ( p != NULL )
	{
		minvalue = (blasint) atol(p);
		if ((minvalue > 0) && ((*M<minvalue) || (*N<minvalue) || (*K<minvalue)))
			use_gpu = 0;

	}
	if ( blas_gpu_info && use_gpu)
	{
		sgemm_gpu = blas_gpu_info(3, "sgemm" , NULL, NULL, NULL);
        	if ( sgemm_gpu )
        	{
			#ifdef DEBUG
				printf("Running on the GPU\n");
			#endif
                	ret = sgemm_gpu(TRANSA, TRANSB, M, N, K, alpha, a, LDA, b, LDB, beta, c, LDC); 
			if ( ret != 0 )
			{
				printf("GPU error: %d\n",ret);
			}
			else
				return;
        	}
	}

	sgemm_cpu = dlsym(RTLD_NEXT, "sgemm_");
       	if ( sgemm_cpu )
       	{
		#ifdef DEBUG
			printf("Running on the CPU\n");
		#endif
               	sgemm_cpu(TRANSA, TRANSB, M, N, K, alpha, a, LDA, b, LDB, beta, c, LDC); 
       	}

	#ifdef DEBUG
		if(fp)
			fclose(fp);
	#endif

}


void cblas_sgemm(int order, int TransA, int TransB,
           blasint m, blasint n, blasint k,
           float alpha,
           float *a, blasint lda,
           float *b, blasint ldb,
           float beta,
           float *c, blasint ldc) 
{

	char transa[2] = "P";
	char transb[2] = "P"; 

	blasint M = m;
	blasint N = n;
	blasint K = k;
	blasint LDA = lda;
	blasint LDB = ldb;
	blasint LDC = ldc;
	float ALPHA = alpha;
	float BETA  = beta;

	#ifdef DEBUG
		FILE *fp;
		fp = fopen("/tmp/wrap.log", "a+");
		fprintf(fp, "Running cblas_sgemm M=%d N=%d K=%d\n", m, n, k);
		printf("Running cblas_sgemm M=%d N=%d K=%d\n", m, n, k);
	#endif

	#ifdef DEBUG
		if(fp)
			fclose(fp);
	#endif

	int info = 0;
	if ( (TransB != 111) && (TransB != 112 )) info=3;
	if ( (TransA != 111) && (TransA != 112 )) info=2;
	if ( (order != 101)  && (order != 102 ) ) info=1;

	if (info > 0)
	{
                printf("** On entry to sgemm_ parameter number %d had a illegal value\n",info);
                return;
	}

	if ( order == 102 ) // ColMajor
	{

		if ( TransA == 111 ) // Notrans
			transa[0] = 'N';
		else
			transa[0] = 'T';

		if ( TransB == 111 )
			transb[0] = 'N';
		else
			transb[0] = 'T';

	}
	else			// RowMajor
	{

		if ( TransA == 111 ) // Notrans
			transa[0] = 'T';
		else
			transa[0] = 'N';

		if ( TransB == 111 )
			transb[0] = 'T';
		else
			transb[0] = 'N';

	}

	sgemm_(transa, transb, &M, &N, &K, &ALPHA, a, &LDA, b, &LDB, &BETA, c, &LDC);

}



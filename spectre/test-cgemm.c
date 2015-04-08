
/***************************************************************************
Copyright (c) 2015, The OpenBLAS Project
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>

int (*cgemm_gpu) (const char * transa,const char * transb,const int *m,const int *n,const int *k,const float *alpha,const float * __restrict__ a,const int *lda,const float *b,const int *ldb,const float *beta,float * c,const int *ldc );

int cgemm_gpu_impl (const char * transa,const char * transb,const int *m,const int *n,const int *k,const float *alpha,const float * __restrict__ a,const int *lda,const float *b,const int *ldb,const float *beta,float * c,const int *ldc );

extern void cgemm_ (const char * transa,const char * transb,const int *m,const int *n,const int *k,const float *alpha,const float * __restrict__ a,const int *lda,const float *b,const int *ldb,const float *beta,float * c,const int *ldc );

void * blas_gpu_info(int level, char *st, int *m, int *n, int *k);

int main (int argc, char *argv[])
{

	struct timeval tv;
	double start,end,timec,timeg;

	int lda,ldb;

	int m=8189;	
	int n=8189;
	int k=8189;	

	char transa='C';
	char transb='C';

	if ( transa == 'N' )
		lda = m;
	else
		lda = k;

	if ( (transb == 'T') || (transb == 'C') )
		ldb = n;
	else
		ldb = k;

	int ldc = m;


	unsigned long i,j;

	float alpha[2] = { 0.2, 0.1 };
	float beta[2] = { 0.3, 2.0 } ;
	float error;

	void *a1,*b1,*c1,*c2;
	float *a,*b,*cg,*cc;

	a1=malloc((size_t) 32768*8192*4);
	b1=malloc((size_t) 32768*8192*4);
	c1=malloc((size_t) 32768*32768*4);
	c2=malloc((size_t) 32768*32768*4);

	a=(float *) a1;
	b=(float *) b1;
	cg=(float *) c1;
	cc=(float *) c2;

	// memset(a,0,2*4096*4096*4);
	for( i = 0; i<m; i++)
	{
		for ( j = 0; j< k; j++)
		{
				*a = (((float) rand() / (float) RAND_MAX) - 0.5) * 1e-1 ;
				// *a = (float) (10*i+k+1) ;
				// *a = (float) 1.0;
				a++;
				*a = (((float) rand() / (float) RAND_MAX) - 0.5) * 1e-1 ;
				// *a = (float) 1.0;
				a++;
		}	
	}
	// memset(b,0,2*4096*4096*4);
	for( i = 0; i<k; i++)
	{
		for( j=0; j<n; j++)
		{
			*b = (((float) rand() / (float) RAND_MAX) - 0.5) * 1e-1;
			// *b = (float) 1.0 ;
			// *b = (float) (i+k+1) ;
			b++;
			*b = (((float) rand() / (float) RAND_MAX) - 0.5) * 1e-1;
			// *b = (float) 1.0 ;
			b++;
		}
	}			

	// #pragma omp parallel for
	// memset(cc,0,2*4096*4096*4);
	// memset(cg,0,2*4096*4096*4);
	for( i = 0; i<n; i++)
	{
		for(j=0; j<m; j++)
		{
			*cc = (float) 1.0;
			*cg = (float) 1.0;
			cc++;
			cg++;
			*cc = (float) 2.0;
			*cg = (float) 2.0;
			cc++;
			cg++;
		}
	}			



	a=(float *) a1;
	b=(float *) b1;
	cg=(float *) c1;
	cc=(float *) c2;
	
	gettimeofday(&tv,NULL);
	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;

	int ret=0;
	int (*cgemm_gpu)();


	cgemm_gpu = blas_gpu_info(3, "cgemm", NULL, NULL, NULL);
	if ( cgemm_gpu )
	{
		ret = cgemm_gpu(&transa,&transb,&m,&n,&k,alpha,a,&lda,b,&ldb,beta,cg,&ldc);
	}
	else
	{
		printf("GPU error\n");
                return(1);

	}
	if ( ret != 0)
	{
		printf("GPU error\n");
		return(1);
	}


	gettimeofday(&tv,NULL);
	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	timeg=end-start;

	gettimeofday(&tv,NULL);
	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;

	cgemm_(&transa, &transb, &m, &n,&k ,alpha, a, &lda, b, &ldb, beta, cc, &ldc);

	gettimeofday(&tv,NULL);
	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	timec=end-start;


	for(i=0; i<m*n*2; i++)
	{
		error = fabs((cc[i] - cg[i]));
		if ( error > 0.1e-4 )
			printf("ERROR: %ld:%.16f	%.16f	%.16f\n",i,error, cg[i],cc[i]);
	}


	double fp =(8.0 * (double) m*n*k  ) * (double) 1.0e-9;
	double gflops=fp / timeg ;
	printf("GPU: %dx%dx%d size\t%10.8f sec\t%10.6f GFlop\t%10.8f GFlops\n",m,n,k,timeg,fp,gflops);
	gflops=fp / timec ;
	printf("CPU: %dx%dx%d size\t%10.8f sec\t%10.6f GFlop\t%10.8f GFlops\n",m,n,k,timec,fp,gflops);
	return(0);

}




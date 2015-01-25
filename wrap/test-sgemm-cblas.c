#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>



void cblas_sgemm(int order, int TransA, int TransB,
           int m, int n, int k,
           float alpha,
           float *a, int lda,
           float *b, int ldb,
           float beta,
           float *c, int ldc);


int main (int argc, char *argv[])
{

	struct timeval tv;
	double start,end,timec,timeg;

	int lda,ldb;

	int m=128;	
	int n=64;
	int k=16;	

	int  order=102;		// ColMajor
	int  transa=111;	// notrans
	int  transb=111;

	if ( transa == 111 )
		lda = m;
	else
		lda = k;

	if ( transb == 111 )
		ldb = k;
	else
		ldb = n;

	int ldc = m;

	int one = 1;


	unsigned long i,j;

	float alpha=2.0;
	float beta=2.0;
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
	int k1;
	// #pragma omp parallel for
	for( i = 0; i<m; i++)
	{
		for ( j = 0; j< k ; j++)
		{
				// *a = (((float) rand() / (float) RAND_MAX) - 0.5) * 1e-1 ;
				*a = (float) (i*10+k+1) *0.01;
				// *a = (float) 1.0;
				a++;
		}	
/*
		for ( j=k ; j<16; j++ )
		{
			*a = (float) 0.0;
			a++;
		}
*/		
	}			
/*
	for ( i=m ; i<128; i++ )
	{
		for ( j=0; j<16; j++)
		{
			*a = (float) 0.0;
			a++;
		}
	}	
*/
	// memset(b,0,2*4096*4096*4);
	// #pragma omp parallel for
	for( i = 0; i<k; i++)
	{
		for( j=0; j<n; j++)
		{
			// *b = (((float) rand() / (float) RAND_MAX) - 0.5) * 1e-1;
			*b = (float) (i*k+1) +i ;
			//*b = (float) (i+k+1) ;
			b++;
		}
/*
		for ( j=n ; j<64; j++ )
		{
			*b = (float) 0.0;
			b++;
		}
*/
	}			

	// #pragma omp parallel for
	for( i = 0; i<n; i++)
	{
		for(j=0; j<m; j++)
		{
			*cc = (float) (i*10+k+3);
			*cg = (float) 0.0;
			cc++;
			cg++;
		}
	}			



	a=(float *) a1;
	b=(float *) b1;
	cg=(float *) c1;
	cc=(float *) c2;

	int ret=0;




	gettimeofday(&tv,NULL);
	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;

	cblas_sgemm(order, transa, transb, m, n, k , alpha, a, lda, b, ldb, beta, cc, ldc);

	gettimeofday(&tv,NULL);
	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	timec=end-start;


	for(i=0; i<m*n; i++)
	{
		// error = fabs((cc[i] - cg[i]));
		// if ( error > 0.1e-4 )
			//printf("ERROR: %d:%.16f	%.16f	%.16f\n",i,error, cg[i],cc[i]);
			printf("%.16f\n",cc[i]);
	}


	double fp =(2.0 * (double) m*n*k  ) * (double) 1.0e-9;
	double gflops=fp / timec ;
	printf("GPU: %dx%dx%d size\t%10.8f sec\t%10.6f GFlop\t%10.8f GFlops\n",m,n,k,timec,fp,gflops);
	return(0);

}




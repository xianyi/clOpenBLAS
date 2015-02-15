#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <math.h>



extern void dgemm_ (const char * transa,const char * transb,const int *m,const int *n,const int *k,const double *alpha,const double * __restrict__ a,const int *lda,const double *b,const int *ldb,const double *beta,double * c,const int *ldc );


int main (int argc, char *argv[])
{

	struct timeval tv;
	double start,end,timec;

	int lda,ldb;

	int m=4096;	
	int n=4096;
	int k=4096;	

	char transa='N';
	char transb='T';

	if ( transa == 'N' )
		lda = m;
	else
		lda = k;

	if ( transb == 'N' )
		ldb = k;
	else
		ldb = n;

	int ldc = m;

	unsigned long i,j;

	double alpha=2.0;
	double beta=2.0;

	void *a1,*b1,*c1,*c2;
	double *a,*b,*cg,*cc;

	a1=malloc((size_t) 32768*8192*4);
	b1=malloc((size_t) 32768*8192*4);
	c1=malloc((size_t) 32768*32768*4);
	c2=malloc((size_t) 32768*32768*4);

	a=(double *) a1;
	b=(double *) b1;
	cg=(double *) c1;
	cc=(double *) c2;

	for( i = 0; i<m; i++)
	{
		for ( j = 0; j< k ; j++)
		{
				*a = (((double) rand() / (double) RAND_MAX) - 0.5) * 1e-1 ;
				//*a = (double) (i+k+1) ;
				// *a = (double) 1.0;
				a++;
		}	
	}			
	for( i = 0; i<k; i++)
	{
		for( j=0; j<n; j++)
		{
			*b = (((double) rand() / (double) RAND_MAX) - 0.5) * 1e-1;
			// *b = (double) (0+1) ;
			//*b = (double) (i+k+1) ;
			b++;
		}
	}			

	for( i = 0; i<n; i++)
	{
		for(j=0; j<m; j++)
		{
			*cc = (double) 0.0;
			*cg = (double) 0.0;
			cc++;
			cg++;
		}
	}			



	a=(double *) a1;
	b=(double *) b1;
	cg=(double *) c1;
	cc=(double *) c2;


	gettimeofday(&tv,NULL);
	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;

	dgemm_(&transa, &transb, &m, &n,&k ,&alpha, a, &lda, b, &ldb, &beta, cc, &ldc);

	gettimeofday(&tv,NULL);
	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	timec=end-start;


	double fp =(2.0 * (double) m*n*k  ) * (double) 1.0e-9;
	double gflops=fp / timec ;
	printf("GPU: %dx%dx%d size\t%10.8f sec\t%10.6f GFlop\t%10.8f GFlops\n",m,n,k,timec,fp,gflops);
	return(0);

}




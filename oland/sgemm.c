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

#include <omp.h>

/************************************* CCOPY *************************************************/
static void sgemm_gpu_ccopy(int M, int N, float *A, int LDA, float *B, blasint LDB , float beta)
{

   if ( beta == (float) 1.0)
   {
	int m1 = M >> 3;
	int m2 = M - m1 * 8;

	int i;

	#pragma omp parallel for num_threads(4)
	for ( i = 0; i < N; i++)	
	{
		float *ap = A + i*LDA;
		float *bp = B + i*LDB;

		float ar[8];

		int j;
	
		for ( j=0; j < m1 ; j++ )
		{

			ar[0] = *(ap + 0);
			ar[1] = *(ap + 1);
			ar[2] = *(ap + 2);
			ar[3] = *(ap + 3);
			ar[4] = *(ap + 4);
			ar[5] = *(ap + 5);
			ar[6] = *(ap + 6);
			ar[7] = *(ap + 7);

			*(bp + 0) += ar[0];
			*(bp + 1) += ar[1];
			*(bp + 2) += ar[2];
			*(bp + 3) += ar[3];
			*(bp + 4) += ar[4];
			*(bp + 5) += ar[5];
			*(bp + 6) += ar[6];
			*(bp + 7) += ar[7];

			ap += 8;
			bp += 8;

		}

		for ( j = 0; j < m2; j++ )
		{
			*(bp + j) += *(ap + j);
		}
	
	}
	return;
   }

   if ( beta == (float) 0.0)
   {
	int m1 = M >> 3;
	int m2 = M - m1 * 8;

	int i;
	#pragma omp parallel for num_threads(2)
	for ( i = 0; i < N; i++)	
	{
		float *ap = A + i*LDA;
		float *bp = B + i*LDB;

		float ar[8];

		int j;
	
		for ( j=0; j < m1 ; j++ )
		{

			ar[0] = *(ap + 0);
			ar[1] = *(ap + 1);
			ar[2] = *(ap + 2);
			ar[3] = *(ap + 3);
			ar[4] = *(ap + 4);
			ar[5] = *(ap + 5);
			ar[6] = *(ap + 6);
			ar[7] = *(ap + 7);

			*(bp + 0) = ar[0];
			*(bp + 1) = ar[1];
			*(bp + 2) = ar[2];
			*(bp + 3) = ar[3];
			*(bp + 4) = ar[4];
			*(bp + 5) = ar[5];
			*(bp + 6) = ar[6];
			*(bp + 7) = ar[7];

			ap += 8;
			bp += 8;

		}

		for ( j = 0; j < m2; j++ )
		{
			*(bp + j) += *(ap + j);
		}
	
	}
	return;
   }

	int m1 = M >> 3;
	int m2 = M - m1 * 8;
	int i;

	#pragma omp parallel for num_threads(4)
	for ( i = 0; i < N; i++)	
	{
		float *ap = A + i*LDA;
		float *bp = B + i*LDB;

		float ar[8];

		int j;
	
		for ( j=0; j < m1 ; j++ )
		{

			ar[0] = *(ap + 0);
			ar[1] = *(ap + 1);
			ar[2] = *(ap + 2);
			ar[3] = *(ap + 3);
			ar[4] = *(ap + 4);
			ar[5] = *(ap + 5);
			ar[6] = *(ap + 6);
			ar[7] = *(ap + 7);

			*(bp + 0) = ar[0] + *(bp + 0) * beta;
			*(bp + 1) = ar[1] + *(bp + 1) * beta;
			*(bp + 2) = ar[2] + *(bp + 2) * beta;
			*(bp + 3) = ar[3] + *(bp + 3) * beta;
			*(bp + 4) = ar[4] + *(bp + 4) * beta;
			*(bp + 5) = ar[5] + *(bp + 5) * beta;
			*(bp + 6) = ar[6] + *(bp + 6) * beta;
			*(bp + 7) = ar[7] + *(bp + 7) * beta;

			ap += 8;
			bp += 8;

		}

		for ( j = 0; j < m2; j++ )
		{
			*(bp + j) = *(ap + j) + *(bp + j) * beta;
		}
	
	}
	return;


}



/************************************* BCOPY *************************************************/

static void sgemm_gpu_bcopy(int M, int N, float *A , blasint LDA, float *B, int PAD_M, int PAD_N)
{
	int i;

	int pad_m = (( M + PAD_M - 1 ) & -PAD_M) - M;
	int pad_n = (( N + PAD_N - 1 ) & -PAD_N) - N;

	int n1 = N/8;
	int n2 = N - n1*8;

	#pragma omp parallel for num_threads(3)
	for ( i=0; i < M ; i++ )
	{

		float *a_ptr = A + i * LDA;
		float *b_ptr = B + i * (N + pad_n);
		
		float *ap = a_ptr;
		float *bp = b_ptr;
		int j;
		float ar[8];

		for ( j=0; j < n1 ; j++ )
		{

			ar[0] = *(ap + 0);
			ar[1] = *(ap + 1);
			ar[2] = *(ap + 2);
			ar[3] = *(ap + 3);
			ar[4] = *(ap + 4);
			ar[5] = *(ap + 5);
			ar[6] = *(ap + 6);
			ar[7] = *(ap + 7);

			*(bp + 0) = ar[0];
			*(bp + 1) = ar[1];
			*(bp + 2) = ar[2];
			*(bp + 3) = ar[3];
			*(bp + 4) = ar[4];
			*(bp + 5) = ar[5];
			*(bp + 6) = ar[6];
			*(bp + 7) = ar[7];

			ap += 8;
			bp += 8;

		}

		for (j=0 ; j < n2 ; j++ )
		{
			*(bp + j) = *(ap + j);
		}

		for ( ; j < n2 + pad_n; j++ )
		{
			*(bp + j) = (float) 0.0;
		}
		// a_ptr += LDA;
		// b_ptr += N + pad_n;
		
	}

	for ( i=M; i < M + pad_m; i++)
	{
		float *b_ptr = B + i * (N + pad_n);

		int j;
		for ( j=0; j < N + pad_n; j++ )
                {
                        *(b_ptr + j) = (float) 0.0;
                }
		// b_ptr += N + pad_n;
	}

}


/************************************* ACOPY *************************************************/

static void sgemm_gpu_acopy(int M, int K, float *A , blasint LDA, float *B, int PAD_M, int PAD_K)
{
	int i;

	int pad_m = (( M + PAD_M - 1 ) & -PAD_M) - M;
	int pad_k = (( K + PAD_K - 1 ) & -PAD_K) - K;


	// float *a_ptr = A;
	// float *b_ptr = B;

	int m1 = M >> 4;
	int m2 = M - m1 * 16;

	#pragma omp parallel for num_threads(3)
	for ( i=0; i < K ; i++ )
	{

		// printf("OMP Thread %d\n", omp_get_thread_num());
		float *a_ptr = A + i * LDA;
		float *b_ptr = B + i * (M + pad_m);
		
		float *ap = a_ptr;
		float *bp = b_ptr;
		int j;
		float ar[16];

		for ( j=0; j < m1 ; j++ )
		{

			ar[0] = *(ap + 0);
			ar[1] = *(ap + 1);
			ar[2] = *(ap + 2);
			ar[3] = *(ap + 3);
			ar[4] = *(ap + 4);
			ar[5] = *(ap + 5);
			ar[6] = *(ap + 6);
			ar[7] = *(ap + 7);
			ar[8] = *(ap + 8);
			ar[9] = *(ap + 9);
			ar[10] = *(ap + 10);
			ar[11] = *(ap + 11);
			ar[12] = *(ap + 12);
			ar[13] = *(ap + 13);
			ar[14] = *(ap + 14);
			ar[15] = *(ap + 15);

			*(bp + 0) = ar[0];
			*(bp + 1) = ar[1];
			*(bp + 2) = ar[2];
			*(bp + 3) = ar[3];
			*(bp + 4) = ar[4];
			*(bp + 5) = ar[5];
			*(bp + 6) = ar[6];
			*(bp + 7) = ar[7];
			*(bp + 8) = ar[8];
			*(bp + 9) = ar[9];
			*(bp + 10) = ar[10];
			*(bp + 11) = ar[11];
			*(bp + 12) = ar[12];
			*(bp + 13) = ar[13];
			*(bp + 14) = ar[14];
			*(bp + 15) = ar[15];

			ap += 16;
			bp += 16;

		}

		for (j=0 ; j < m2 ; j++ )
		{
			*(bp + j) = *(ap + j);
			// printf("j=%d	%f	%f\n",j,*(bp+j),*(ap+j));
		}

		for ( ; j < m2 + pad_m; j++ )
		{
			*(bp + j) = (float) 0.0;
			//  printf("j=%d	%f	%f\n",j,*(bp+j),0.0);
		}
		// a_ptr += LDA;
		// b_ptr += M + pad_m;
		
	}

	for ( i=K; i < K + pad_k; i++)
	{
		float *b_ptr = B + i * (M + pad_m);

		int j;
		for ( j=0; j < M + pad_m; j++ )
                {
                        *(b_ptr + j) = (float) 0.0;
                }
		// b_ptr += M + pad_m;
	}

}


/************************************* ATCOPY *************************************************/
/*
static void sgemm_gpu_atcopy(int M, int K, float *A , blasint LDA, float *B, int PAD_M, int PAD_K)
{
	int i;

	int pad_m = (( M + PAD_M - 1 ) & -PAD_M) - M;

	int mp = M + pad_m;


	float *a_ptr = A;
	float *b_ptr = B;

	for ( i=0; i < (M/2)*2 ; i+=2 )
	{

		
		float *ap = a_ptr;
		float *ap1 = a_ptr + LDA;
		float *bp = b_ptr;
		float *bp1 = b_ptr +1;
		int j;
		float ar[8];
		
		int mp0 = 0;
		int mp1 = mp;
		int mp2 = mp1+mp;
		int mp3 = mp2+mp;

		for (j=0 ; j < (K/4)*4 ; j+=4 )
		{

			ar[0] = *(ap);  
			ar[1] = *(ap+1);  
			ar[2] = *(ap+2);  
			ar[3] = *(ap+3);  

			ar[4] = *(ap1);
			ar[5] = *(ap1+1);
			ar[6] = *(ap1+2);
			ar[7] = *(ap1+3);

			*(bp  + mp0) = ar[0];
			*(bp1 + mp0) = ar[4];
			*(bp  + mp1) = ar[1];
			*(bp1 + mp1) = ar[5];
			*(bp  + mp2) = ar[2];
			*(bp1 + mp2) = ar[6];
			*(bp  + mp3) = ar[3];
			*(bp1 + mp3) = ar[7];

			mp0 += 4 * mp;
			mp1 = mp0 + mp;
			mp2 = mp1 + mp;
			mp3 = mp2 + mp;
			ap  += 4;
			ap1 += 4;
		}

		for ( ; j < K ; j++ )
		{

			ar[0] = *(ap);  
			ar[4] = *(ap1);  

			*(bp + mp0 )        = ar[0];
			*(bp + mp0 +1)      = ar[4];

			mp0 += mp;
			ap++;
			ap1++;
		}

		a_ptr += 2*LDA;
		b_ptr += 2;
	}

	for ( ; i < M ; i++ )
	{

		
		float *ap = a_ptr;
		float *bp = b_ptr;
		int j;
		float ar[8];
		
		int mp0 = 0;
		int mp1 = mp;
		int mp2 = mp1+mp;
		int mp3 = mp2+mp;

		for (j=0 ; j < (K/4)*4 ; j+=4 )
		{

			ar[0] = *(ap);  
			ar[1] = *(ap+1);  
			ar[2] = *(ap+2);  
			ar[3] = *(ap+3);  

			*(bp  + mp0) = ar[0];
			*(bp  + mp1) = ar[1];
			*(bp  + mp2) = ar[2];
			*(bp  + mp3) = ar[3];

			mp0 += 4 * mp;
			mp1 = mp0 + mp;
			mp2 = mp1 + mp;
			mp3 = mp2 + mp;
			ap  += 4;
		}

		for ( ; j < K ; j++ )
		{

			ar[0] = *(ap);  

			*(bp + mp0 )        = ar[0];

			mp0 += mp;
			ap++;
		}

		a_ptr += 2*LDA;
		b_ptr += 2;
	}

}
*/

static void sgemm_gpu_btcopy(int M, int K, float *A , blasint LDA, float *B, int PAD_M, int PAD_K)
{
	unsigned long i=0;

	int pad_m = (( M + PAD_M - 1 ) & -PAD_M) - M;

	int mp = M + pad_m;


	float *a_ptr = A;
	float *b_ptr = B;

	// #pragma omp parallel for num_threads(3)
	for ( i=0; i < (M/4)*4 ; i+=4 )
	{

		// float *a_ptr = A + i * (unsigned long) LDA;
		// float *b_ptr = B + i * 4;
	
		
		float *ap = a_ptr;
		float *ap1 = ap  + LDA;
		float *ap2 = ap1 + LDA;
		float *ap3 = ap2 + LDA;

		float *bp = b_ptr;
		float *bp1 = b_ptr +1;
		float *bp2 = b_ptr +2;
		float *bp3 = b_ptr +3;

		int j;
		float ar[16];
		
		int mp0 = 0;
		int mp1 = mp;
		int mp2 = mp1+mp;
		int mp3 = mp2+mp;

		for (j=0 ; j < (K/4)*4 ; j+=4 )
		{

			ar[0] = *(ap);  
			ar[1] = *(ap+1);  
			ar[2] = *(ap+2);  
			ar[3] = *(ap+3);  

			ar[4] = *(ap1);
			ar[5] = *(ap1+1);
			ar[6] = *(ap1+2);
			ar[7] = *(ap1+3);

			ar[8]  = *(ap2);
			ar[9]  = *(ap2+1);
			ar[10] = *(ap2+2);
			ar[11] = *(ap2+3);

			ar[12] = *(ap3);
			ar[13] = *(ap3+1);
			ar[14] = *(ap3+2);
			ar[15] = *(ap3+3);

			*(bp  + mp0) = ar[0];
			*(bp1 + mp0) = ar[4];
			*(bp2 + mp0) = ar[8];
			*(bp3 + mp0) = ar[12];
			*(bp  + mp1) = ar[1];
			*(bp1 + mp1) = ar[5];
			*(bp2 + mp1) = ar[9];
			*(bp3 + mp1) = ar[13];
			*(bp  + mp2) = ar[2];
			*(bp1 + mp2) = ar[6];
			*(bp2 + mp2) = ar[10];
			*(bp3 + mp2) = ar[14];
			*(bp  + mp3) = ar[3];
			*(bp1 + mp3) = ar[7];
			*(bp2 + mp3) = ar[11];
			*(bp3 + mp3) = ar[15];

			mp0 += 4 * mp;
			mp1 = mp0 + mp;
			mp2 = mp1 + mp;
			mp3 = mp2 + mp;
			ap  += 4;
			ap1 += 4;
			ap2 += 4;
			ap3 += 4;
		}

		for ( ; j < K ; j++ )
		{

			ar[0]  = *(ap);  
			ar[4]  = *(ap1);  
			ar[8]  = *(ap2);  
			ar[12] = *(ap3);  

			*(bp + mp0 )        = ar[0];
			*(bp + mp0 +1)      = ar[4];
			*(bp + mp0 +2)      = ar[8];
			*(bp + mp0 +3)      = ar[12];

			mp0 += mp;
			ap++;
			ap1++;
			ap2++;
			ap3++;
		}

		a_ptr += 4*LDA;
		b_ptr += 4;
	}

	for ( ; i < M ; i++ )
	{

		//float *a_ptr = A + i * LDA;
		//float *b_ptr = B + i ;
		
		float *ap = a_ptr;
		float *bp = b_ptr;
		int j;
		float ar[8];
		
		int mp0 = 0;
		int mp1 = mp;
		int mp2 = mp1+mp;
		int mp3 = mp2+mp;

		for (j=0 ; j < (K/4)*4 ; j+=4 )
		{

			ar[0] = *(ap);  
			ar[1] = *(ap+1);  
			ar[2] = *(ap+2);  
			ar[3] = *(ap+3);  

			*(bp  + mp0) = ar[0];
			*(bp  + mp1) = ar[1];
			*(bp  + mp2) = ar[2];
			*(bp  + mp3) = ar[3];

			mp0 += 4 * mp;
			mp1 = mp0 + mp;
			mp2 = mp1 + mp;
			mp3 = mp2 + mp;
			ap  += 4;
		}

		for ( ; j < K ; j++ )
		{

			ar[0] = *(ap);  

			*(bp + mp0 )        = ar[0];

			mp0 += mp;
			ap++;
		}

		a_ptr += 1*LDA;
		b_ptr += 1;
	}

}




static int sgemm_gpu_simple(char *TRANSA, char *TRANSB, blasint *M, blasint *N, blasint *K, float *ALPHA, float *A, blasint *LDA, float *B, blasint *LDB, float *BETA, float *C, blasint *LDC)
{

	int ret;

	#ifdef PROFILE
        	struct timeval tv;
        	double start,end,time;
	#endif

	double ktime = 0.0;

	if ( have_gpu_context == 0 )
	{
		printf("GPU Error\n");
		return(1);
	}

	#ifdef PROFILE
		printf("BEGIN create program\n");
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

        ret = create_gpu_program_nonunified(&gpu, "gemm",  (size_t) ALLOC_SIZE);
        if ( ret )
	{
		have_gpu_context = 0;
		printf("GPU Error\n");
                return(1);

	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
		time=end-start;
		printf("END create program\n");
		printf("OpenCL create program summary:\t\t%f sec\n", time);
	#endif

	int info = 0;
	int transa = -1;
	int transb = -1;
	char transA = toupper(*TRANSA);
	char transB = toupper(*TRANSB);

	if ( transA == 'N' ) transa = 0;
	if ( transA == 'T' ) transa = 1;
	if ( transB == 'N' ) transb = 0;
	if ( transB == 'T' ) transb = 1;

	if ( transb < 0 ) info = 2;
	if ( transa < 0 ) info = 1;

	if ( info > 0 )
	{
		printf("** On entry to sgemm_ parameter number %d had a illegal value\n",info);
		return(2);
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
		return(2);
	}


	void *bcopy_ptr[SGEMM_N_BUFFERS + 1];
	int i1;
	for (i1 = 0; i1 < SGEMM_N_BUFFERS+1; i1++)
		bcopy_ptr[i1] = gpu.hB + i1 * GALLOC_SIZE_B;

	int SGEMM_M_MAX_RUN;
	int SGEMM_N_MAX_RUN;
	int SGEMM_K_MAX_RUN = SGEMM_K_MAX;


	if ( transA == 'N' )
		SGEMM_M_MAX_RUN = SGEMM_M_MAX;
	else
		SGEMM_M_MAX_RUN = 1536;
	
	if ( transB == 'N' )
		SGEMM_N_MAX_RUN = 1536;
	else
		SGEMM_N_MAX_RUN = SGEMM_N_MAX;
	

	int bcopy = 1;
	int acopy = 1;

	#ifdef PROFILE
		ktime = 0.0;
	#endif

	float beta = *BETA;

	int sgemm_m_run;
	int sgemm_n_run;
	int sgemm_k_run;

	int sgemm_k = SGEMM_K_MAX_RUN;
	int l=0;

	while ( sgemm_k == SGEMM_K_MAX_RUN )
	{
		
		int saved_bcopy=-1;

		if ( l * SGEMM_K_MAX_RUN + sgemm_k > *K )
		{
			sgemm_k = *K % SGEMM_K_MAX_RUN;
			if ( sgemm_k == 0 )
			{
				break;
			}
		}

		sgemm_k_run = ( sgemm_k + SGEMM_PAD_K - 1 ) & -SGEMM_PAD_K;

		int sgemm_m = SGEMM_M_MAX_RUN;
		int i=0;

		while ( sgemm_m == SGEMM_M_MAX_RUN )
		{
			if ( i * SGEMM_M_MAX_RUN + sgemm_m > *M )
			{
				sgemm_m = *M % SGEMM_M_MAX_RUN;
				if ( sgemm_m == 0 )
				{
					break;
				}	
			}

			sgemm_m_run = ( sgemm_m + SGEMM_PAD_M - 1 ) & -SGEMM_PAD_M;

			float *a_ptr;

			if ( transA == 'N' )
			{
				a_ptr = A + l* SGEMM_K_MAX_RUN * *LDA + i * SGEMM_M_MAX_RUN;

				#ifdef PROFILE
					printf("----------------------------------------------------------------------------------\n");
        				gettimeofday(&tv,NULL);
        				start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
				#endif

				sgemm_gpu_acopy(sgemm_m, sgemm_k, a_ptr, *LDA, (float*) gpu.hA, SGEMM_PAD_M, SGEMM_PAD_K);

				#ifdef PROFILE
        				gettimeofday(&tv,NULL);
        				end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
					time=end-start;
					printf("Sgemm acopy:\t\t\t\t%f sec\n", time);
				#endif

				#ifdef PROFILE
        				gettimeofday(&tv,NULL);
        				start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
				#endif

				clEnqueueWriteBuffer(gpu.command_queue, gpu.A, CL_FALSE, 0, sgemm_m_run * sgemm_k_run *sizeof(float), gpu.hA, 0, NULL, NULL);

				#ifdef PROFILE
        				gettimeofday(&tv,NULL);
        				end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
					time=end-start;
					printf("Prof: enqueue A:\t\t\t%f sec\n", time);
				#endif

			}
			else
			{
				a_ptr = A + i * SGEMM_M_MAX_RUN * *LDA + l * SGEMM_K_MAX_RUN;

				#ifdef PROFILE
					printf("----------------------------------------------------------------------------------\n");
        				gettimeofday(&tv,NULL);
        				start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
				#endif

				memset(gpu.hA, 0, (size_t) sgemm_m_run * sgemm_k_run * sizeof(float));
				sgemm_gpu_btcopy(sgemm_m, sgemm_k, a_ptr, *LDA, (float*) gpu.hA, SGEMM_PAD_M, SGEMM_PAD_K);

				#ifdef PROFILE
        				gettimeofday(&tv,NULL);
        				end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
					time=end-start;
					printf("Sgemm atcopy:\t\t\t\t%f sec\n", time);
				#endif

				#ifdef PROFILE
        				gettimeofday(&tv,NULL);
        				start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
				#endif

				clEnqueueWriteBuffer(gpu.command_queue, gpu.A, CL_FALSE, 0, sgemm_m_run * sgemm_k_run *sizeof(float), gpu.hA, 0, NULL, NULL);

				#ifdef PROFILE
        				gettimeofday(&tv,NULL);
        				end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
					time=end-start;
					printf("Prof: enqueue A:\t\t\t%f sec\n", time);
				#endif


			}

			int sgemm_n = SGEMM_N_MAX_RUN;
			int j=0;
			acopy = 1;

			int curr_bcopy = 0;

			while ( sgemm_n == SGEMM_N_MAX_RUN )
			{

				if ( j * SGEMM_N_MAX_RUN + sgemm_n > *N )
				{
					sgemm_n = *N % SGEMM_N_MAX_RUN;
					if ( sgemm_n == 0 )
					{
						break;
					}
				}

				sgemm_n_run = ( sgemm_n + SGEMM_PAD_N - 1 ) & -SGEMM_PAD_N;

				float *c_ptr = C + j * SGEMM_N_MAX_RUN * *LDC + i * SGEMM_M_MAX_RUN; 
				float *b_ptr;

				if ( transB == 'N' )
				{
					b_ptr = B + j* SGEMM_N_MAX_RUN * *LDB + l*SGEMM_K_MAX_RUN;

					if ( curr_bcopy > saved_bcopy )
					{

						#ifdef PROFILE
        						gettimeofday(&tv,NULL);
        						start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
						#endif

						if ( curr_bcopy < SGEMM_N_BUFFERS )
						{
							memset(bcopy_ptr[curr_bcopy], 0, (size_t) sgemm_n_run * sgemm_k_run * sizeof(float));
							sgemm_gpu_btcopy(sgemm_n, sgemm_k, b_ptr, *LDB, (float*) bcopy_ptr[curr_bcopy], SGEMM_PAD_N, SGEMM_PAD_K);
							saved_bcopy++;
						}
						else
						{
							memset(bcopy_ptr[SGEMM_N_BUFFERS], 0, (size_t) sgemm_n_run * sgemm_k_run * sizeof(float));
							sgemm_gpu_btcopy(sgemm_n, sgemm_k, b_ptr, *LDB, (float*) bcopy_ptr[SGEMM_N_BUFFERS], SGEMM_PAD_N, SGEMM_PAD_K);

						}

						#ifdef PROFILE
        						gettimeofday(&tv,NULL);
        						end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
							time=end-start;
							printf("Sgemm btcopy:\t\t\t\t%f sec\n", time);
						#endif
					}

					#ifdef PROFILE
        					gettimeofday(&tv,NULL);
        					start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
					#endif

					if ( curr_bcopy < SGEMM_N_BUFFERS )
						clEnqueueWriteBuffer(gpu.command_queue, gpu.B, CL_FALSE, 0, sgemm_n_run * sgemm_k_run *sizeof(float), bcopy_ptr[curr_bcopy], 0, NULL, NULL);
					else
						clEnqueueWriteBuffer(gpu.command_queue, gpu.B, CL_FALSE, 0, sgemm_n_run * sgemm_k_run *sizeof(float), bcopy_ptr[SGEMM_N_BUFFERS], 0, NULL, NULL);

					#ifdef PROFILE
        					gettimeofday(&tv,NULL);
        					end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
						time=end-start;
						printf("Prof: enqueue B:\t\t\t%f sec\n", time);
					#endif



				}
				else
				{
					b_ptr = B + l* SGEMM_K_MAX_RUN * *LDB + j*SGEMM_N_MAX_RUN;

					if ( curr_bcopy > saved_bcopy )
					{

						#ifdef PROFILE
        						gettimeofday(&tv,NULL);
        						start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
						#endif

						if ( curr_bcopy < SGEMM_N_BUFFERS )
						{
							sgemm_gpu_bcopy(sgemm_k, sgemm_n, b_ptr, *LDB, (float*) bcopy_ptr[curr_bcopy], SGEMM_PAD_K, SGEMM_PAD_N);
							saved_bcopy++;
						}
						else
						{
							sgemm_gpu_bcopy(sgemm_k, sgemm_n, b_ptr, *LDB, (float*) bcopy_ptr[SGEMM_N_BUFFERS], SGEMM_PAD_K, SGEMM_PAD_N);

						}

						#ifdef PROFILE
        						gettimeofday(&tv,NULL);
        						end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
							time=end-start;
							printf("Sgemm bcopy:\t\t\t\t%f sec\n", time);
						#endif
					}

					#ifdef PROFILE
        					gettimeofday(&tv,NULL);
        					start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
					#endif

					if ( curr_bcopy < SGEMM_N_BUFFERS )
						clEnqueueWriteBuffer(gpu.command_queue, gpu.B, CL_FALSE, 0, sgemm_n_run * sgemm_k_run *sizeof(float), bcopy_ptr[curr_bcopy], 0, NULL, NULL);
					else
						clEnqueueWriteBuffer(gpu.command_queue, gpu.B, CL_FALSE, 0, sgemm_n_run * sgemm_k_run *sizeof(float), bcopy_ptr[SGEMM_N_BUFFERS], 0, NULL, NULL);

					#ifdef PROFILE
       						gettimeofday(&tv,NULL);
       						end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
						time=end-start;
						printf("Prof: enqueue B:\t\t\t%f sec\n", time);
					#endif


				}

				// printf("ACOPY: %d	BCOPY: %d\n", acopy,bcopy);
				ret = sgemm_gpu_kernel(&gpu, sgemm_m_run, sgemm_n_run, sgemm_k_run, *ALPHA, acopy, bcopy, &ktime);
				if ( ret != CL_SUCCESS )
				{
					
					printf("GPU Error\n");
					release_gpu_program(&gpu);
					return(1);
				}

				#ifdef PROFILE
        				gettimeofday(&tv,NULL);
        				start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
				#endif
				sgemm_gpu_ccopy(sgemm_m, sgemm_n, (float*) gpu.hC, sgemm_m_run , c_ptr, *LDC, beta);
				#ifdef PROFILE
        				gettimeofday(&tv,NULL);
        				end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
					time=end-start;
					printf("Sgemm ccopy:\t\t\t\t%f sec\n\n", time);
				#endif


				acopy = 0;
				j++;
				curr_bcopy++;

			}
			i++;

		}
		beta = (float) 1.0;
		l++;
	}

	#ifdef PROFILE
		printf("----------------------------------------------------------------------------------\n");
		printf("OpenCL Kernel time sum:\t\t\t%f sec\n", ktime);
		printf("OpenCL GFLops:\t\t\t\t%f\n", (double) *K * (double) *M * (double) *N * (double) 2.0 / ktime  * (double) 1e-9);
		printf("----------------------------------------------------------------------------------\n");
	#endif
	release_gpu_program(&gpu);
	// destroy_gpu_context(&gpu);
	return(0);

}


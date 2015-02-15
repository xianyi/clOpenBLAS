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

void dgemm_(char *TRANSA, char *TRANSB, blasint *M, blasint *N, blasint *K, double *alpha, double *a, blasint *LDA, double *b, blasint *LDB, double *beta, double *c, blasint *LDC )
{
	int ret;
	int (*dgemm_gpu)();
	void (*dgemm_cpu)();

	#ifdef DEBUG
		FILE *fp1;
		fp1 = fopen("/tmp/wrap.log", "a+");
		fprintf(fp1, "dgemm called M=%d N=%d K=%d\n", *M, *N, *K);
		printf("dgemm called M=%d N=%d K=%d\n", *M, *N, *K);
		if (fp1)
			fclose(fp1);
	#endif

        char *p;

	blasint minvalue;
	int use_gpu = 1;

	p = getenv("DGEMM_GPU_MINSIZE");
	if ( p != NULL )
	{
		minvalue = (blasint) atol(p);
		if ( minvalue < 0 )
			use_gpu = 0;
		else
			if ((minvalue > 0) && ((*M<minvalue) || (*N<minvalue) || (*K<minvalue)))
				use_gpu = 0;

	}

/*

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
                printf("** On entry to dgemm_ parameter number %d had a illegal value\n",info);
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
                printf("** On entry to dgemm_ parameter number %d had a illegal value\n",info);
                return;
        }

*/
	#ifdef DEBUG
		FILE *fp;
		fp = fopen("/tmp/wrap.log", "a+");
		fprintf(fp, "Running dgemm M=%d N=%d K=%d\n", *M, *N, *K);
		printf("Running dgemm M=%d N=%d K=%d\n", *M, *N, *K);
	#endif

	void * handle = NULL;
	void * (*blas_gpu_info)() = NULL;

	if ( use_gpu )
	{
        	p=getenv("OPENBLAS_GPU_LIB");
        	if ( p == NULL)
        	{
                	#ifdef DEBUG
                        	fprintf(stderr, "Variable OPENBLAS_GPU_LIB not found\n");
                	#endif
        	}

        	handle = dlopen( p, RTLD_LAZY);
        	if ( handle == NULL )
        	{
                	#ifdef DEBUG
                        	fprintf(stderr,"%s\n",dlerror());
                	#endif
        	}
		else
		{
        		blas_gpu_info = dlsym(handle, "blas_gpu_info");
        		if ( blas_gpu_info == NULL )
        		{
                		#ifdef DEBUG
                        		fprintf(stderr, "%s\n",dlerror());
                		#endif
			}

        	}
	}
	if ( blas_gpu_info && use_gpu)
	{

		p=getenv("OPENBLAS_PRECISION");
		if ( p != NULL )
		{
			if ( !strncasecmp(p,"mixed", 8) )
				dgemm_gpu = (blas_gpu_info)(3, "dsgemm" , M, N, K);
			else
				dgemm_gpu = (blas_gpu_info)(3, "dgemm" , M, N, K);
				
		}
		else
			dgemm_gpu = (blas_gpu_info)(3, "dgemm" , M, N, K);

        	if ( dgemm_gpu )
        	{
			#ifdef DEBUG
				printf("Running on the GPU\n");
			#endif
                	ret = dgemm_gpu(TRANSA, TRANSB, M, N, K, alpha, a, LDA, b, LDB, beta, c, LDC); 
			if ( ret != 0 )
			{
				printf("GPU error: %d\n",ret);
			}
			else
				return;
        	}
	}

	void * ohandle = dlopen("libopenblas.so",RTLD_LAZY);

	if ( ohandle)
		dgemm_cpu = dlsym(ohandle, "dgemm_");
	else
		dgemm_cpu = dlsym(RTLD_NEXT, "dgemm_");

       	if ( dgemm_cpu )
       	{
		#ifdef DEBUG
			printf("Running on the CPU\n");
		#endif
               	dgemm_cpu(TRANSA, TRANSB, M, N, K, alpha, a, LDA, b, LDB, beta, c, LDC); 
       	}

	#ifdef DEBUG
		if(fp)
			fclose(fp);
	#endif

}


void cblas_dgemm(int order, int TransA, int TransB,
           blasint m, blasint n, blasint k,
           double alpha,
           double *a, blasint lda,
           double *b, blasint ldb,
           double beta,
           double *c, blasint ldc) 
{

	char transa[2] = "P";
	char transb[2] = "P"; 

	blasint M = m;
	blasint N = n;
	blasint K = k;
	blasint LDA = lda;
	blasint LDB = ldb;
	blasint LDC = ldc;
	double ALPHA = alpha;
	double BETA  = beta;

	#ifdef DEBUG
		FILE *fp;
		fp = fopen("/tmp/wrap.log", "a+");
		fprintf(fp, "Running cblas_dgemm M=%d N=%d K=%d\n", m, n, k);
		printf("Running cblas_dgemm M=%d N=%d K=%d\n", m, n, k);
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
                printf("** On entry to dgemm_ parameter number %d had a illegal value\n",info);
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

	dgemm_(transa, transb, &M, &N, &K, &ALPHA, a, &LDA, b, &LDB, &BETA, c, &LDC);

}



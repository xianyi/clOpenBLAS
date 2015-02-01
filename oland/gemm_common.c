
/*
static void save_gpu_kernel(struct gpu_context *gpu, char * filename )
{

	cl_uint numDevices = 0;
	cl_uint ret=0;
	int i;

	ret |= clGetProgramInfo(gpu->program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint), &numDevices, NULL);

	cl_device_id *devices = (cl_device_id *) malloc( (size_t) numDevices * sizeof(cl_device_id) );	

	ret |= clGetProgramInfo(gpu->program, CL_PROGRAM_DEVICES, sizeof(cl_device_id) * numDevices, devices, NULL);

	size_t *pSizes = (size_t *) malloc( (size_t) numDevices * sizeof(size_t) );

	unsigned char **pBinaries = malloc( (size_t) numDevices * sizeof(char *));
	
	for ( i= 0; i<numDevices; i++ )
	{
		pBinaries[i] = (unsigned char *) malloc( (size_t) pSizes[i] );
	}

	ret |= clGetProgramInfo(gpu->program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices, pBinaries, NULL);

	for ( i=0; i<numDevices; i++ )
	{

		if ( devices[i] == gpu->device_id )
		{
			FILE *fp = fopen(filename, "w");
			if ( fp != NULL )
			{
				fwrite(pBinaries[i], 1, pSizes[i], fp);
				fclose(fp);
			}
		}
		free (pBinaries[i]);
	}
	free (pBinaries);
	free (devices);
	free (pSizes);

}

*/
static void replaceBlanks( char *s)
{
	char *p = s;
	while ( *p )
	{
		if ( *p == ' ')
			*p = '_';
		else
		    	*p = tolower(*p);
		p++;
	}
}

static void destroy_gpu_context(struct gpu_context *gpu)
{

	if (gpu -> command_queue)
	{
		clFlush(gpu -> command_queue);
		clFinish(gpu -> command_queue);
	}

	if (gpu -> kernel)
		clReleaseKernel(gpu -> kernel);

	if (gpu -> program)
		clReleaseProgram(gpu -> program);

	if (gpu -> A)
		clReleaseMemObject(gpu -> A);

	if (gpu -> B)
		clReleaseMemObject(gpu -> B);

	if (gpu -> C)
		clReleaseMemObject(gpu -> C);

	if (gpu -> command_queue)
		clReleaseCommandQueue(gpu -> command_queue);

	if (gpu -> context)
		clReleaseContext(gpu -> context);

	if (gpu -> hA)
		free(gpu->hA);

	if (gpu -> hB)
		free(gpu->hB);

	if (gpu -> hC)
		free(gpu->hC);

	memset(gpu, 0, sizeof(struct gpu_context));
	have_gpu_context = 0;
}


static void release_gpu_program(struct gpu_context *gpu)
{

	if (gpu -> command_queue)
	{
		clFlush(gpu -> command_queue);
		clFinish(gpu -> command_queue);
	}

	if (gpu -> kernel)
	{
		clReleaseKernel(gpu -> kernel);
		gpu -> kernel = NULL;
	}
	if (gpu -> program)
	{
		clReleaseProgram(gpu -> program);
		gpu -> program = NULL;
	}
	if (gpu -> A)
	{
		clReleaseMemObject(gpu -> A);
		gpu -> A = NULL;
	}

	if (gpu -> B)
	{
		clReleaseMemObject(gpu -> B);
		gpu -> B = NULL;
	}
	if (gpu -> C)
	{
		clReleaseMemObject(gpu -> C);
		gpu -> C = NULL;
	}
	if (gpu -> command_queue)
	{
		clReleaseCommandQueue(gpu -> command_queue);
		gpu -> command_queue = NULL;
	}

	if (gpu -> hA)
	{
		free(gpu->hA);
		gpu -> hA = NULL;
	}

	if (gpu -> hB)
	{
		free(gpu->hB);
		gpu -> hB = NULL;
	}

	if (gpu -> hB)
	{
		free(gpu->hB);
		gpu -> hB = NULL;
	}


}


static int build_gpu_program(struct gpu_context *gpu, char *func)
{
 
	cl_int ret;

	#ifdef PROFILE
	struct timeval tv;
        double start,end,time;
	#endif

	char *p;

	size_t valueSize;
	char value[2048];
	char binvalue[2048];
	char *source = NULL;
	int have_binary = 0;

	source = (char *) malloc( (size_t) 128*1024*1024);

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	p = getenv("OPENBLAS_CL_DIR");
	if ( p != NULL )
	{
		strncpy(value,p,1024);
		strcat(value,"/");
	}
	else
	{
		strcpy(value,"");
	}
	strcat(value,func);
	strcat(value,"_kernel_");
	p = getenv("OPENBLAS_CL_KERNEL");
	if ( p )
		strncat(value,p,64);
	else	
		strncat(value,DEFAULT_KERNEL,64);
	strncpy(binvalue, value, 1024);
	strcat(value,".cl");
	strcat(binvalue,".clbin");

	#ifdef DEBUG
		printf("Kernel source: %s\n", value);
		printf("Kernel binary: %s\n", binvalue);

	#endif

	FILE *fp,*fpbin;
	
	fpbin = fopen(binvalue, "r");
	if ( fpbin != NULL )
	{
		valueSize = fread( source , 1, 64*1024*1024, fpbin);
		fclose(fpbin);
		gpu->program = clCreateProgramWithBinary(gpu->context, 1, &gpu->device_id, (const size_t *) &valueSize, (const unsigned char ** ) &source, NULL, &ret);
		if ( ret == CL_SUCCESS )
		{
			#ifdef DEBUG
				printf("Loaded program from binary\n");
			#endif
			have_binary = 1;
		}
	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL search for source or binary:\t%f sec\n", time);
	#endif


	if ( have_binary == 0 )
	{

		#ifdef PROFILE
        		gettimeofday(&tv,NULL);
        		start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
		#endif

		fp = fopen(value, "r");
		if ( fp == NULL )
		{
			#ifdef DEBUG
				printf("Error: file %s not found\n",value);
			#endif
			if ( source )
				free(source);
			return(1);
		}
		valueSize = fread( source , 1, 512*1024, fp);
		fclose(fp);
		*(source + valueSize +1) = '\0';

		#ifdef PROFILE
        		gettimeofday(&tv,NULL);
        		end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        		time=end-start;
        		printf("OpenCL read source:\t\t\t%f sec\n", time);
		#endif

		#ifdef PROFILE
        		gettimeofday(&tv,NULL);
        		start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
		#endif

	
		gpu->program = clCreateProgramWithSource(gpu->context, 1, (const char ** ) &source, (const size_t *) &valueSize, &ret);
		if ( ret != CL_SUCCESS )
		{
			#ifdef DEBUG
				printf("Error: Create Program with source\n");
			#endif
			if ( source )
				free(source);
			return(1);
		}

		#ifdef PROFILE
        		gettimeofday(&tv,NULL);
        		end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        		time=end-start;
        		printf("OpenCL create program from source:\t%f sec\n", time);
		#endif

		#ifdef PROFILE
        		gettimeofday(&tv,NULL);
        		start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
		#endif

		ret = clBuildProgram(gpu->program, 1, &gpu->device_id, NULL, NULL, NULL);

		if ( ret != CL_SUCCESS )
		{
			#ifdef DEBUG
				printf("Error: build program %d\n",ret);
				int status;
				size_t logSize;
				char *log;
				clGetProgramBuildInfo(gpu->program, gpu->device_id, CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &status, NULL);
				clGetProgramBuildInfo(gpu->program, gpu->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
				log = (char *) malloc( (size_t) logSize+1 );
				clGetProgramBuildInfo(gpu->program, gpu->device_id, CL_PROGRAM_BUILD_LOG, logSize+1, log, NULL);
				printf("status=%d \n\n%s\n",status, log);
				free(log);
			#endif
			if ( source )
				free(source);
			return(1);
		}	 
		
		#ifdef PROFILE
        		gettimeofday(&tv,NULL);
        		end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        		time=end-start;
        		printf("OpenCL build program:\t\t\t%f sec\n", time);
		#endif


		// save_gpu_kernel(gpu, binvalue);
			
	}

	if ( source )
		free(source);
	return(0);


}


static int create_gpu_context(struct gpu_context *gpu)
{

	int i=0,j=0;

	#ifdef PROFILE
	struct timeval tv;
        double start,end,time;
	#endif

	cl_uint num_devices=0;
    	cl_uint num_platforms=0;
	cl_int ret;

	cl_platform_id platforms[4];
	cl_device_id devices[4];
	cl_device_type type;

	char *p1;

	char value[2048];
	size_t valueSize;

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	ret = clGetPlatformIDs(4, platforms, &num_platforms);
	if ( ret != CL_SUCCESS || num_platforms<1)
	{
		#ifdef DEBUG
			printf("Error: NO Platforms found\n");
		#endif
		goto NO_SUCCESS;
	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL search platforms:\t\t%f sec\n", time);
	#endif

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	p1 = getenv("OPENBLAS_CL_DEVICE");
	if ( p1 == NULL )
		p1 = DEFAULT_DEVICE;

	int found1 = 0;

	for ( i = 0; i<num_platforms; i++ )
	{
		ret = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_ALL, 4, devices, &num_devices);
		if ( ret == CL_SUCCESS && num_devices > 0 )
		{
			
			for ( j=0; j<num_devices; j++ )
			{

				ret = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, 0, NULL, &valueSize);
				ret = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, valueSize , &type, NULL);
				if ( type == CL_DEVICE_TYPE_CPU )
				{
					strcpy(value,"cpu");
				}
				else
				{
					ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &valueSize);
					ret = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, valueSize, value, NULL);
					replaceBlanks(value);
				}
				#ifdef DEBUG
					printf("Device: %d:%d:%s\n",i,j,value);
				#endif
				
				if ( !strncasecmp(value,p1,strlen(p1)))
				{
					gpu->platform = i;	
					gpu->device   = j;	
					gpu->platform_id = platforms[i];
					gpu->device_id   = devices[j];
					strncpy(gpu->device_name,p1,64);
					found1 = 1;
					break;
				}
				

			}
			if ( found1) break;
		}
		if ( found1 ) break;
	}
	
	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL search device:\t\t\t%f sec\n", time);
	#endif

				 
	if ( found1 == 0 )
	{
		#ifdef DEBUG 
			printf("Error: Device %s not found\n",p1);
		#endif
		goto NO_SUCCESS;
	}

	#ifdef DEBUG
		printf("Num_Devices: %d\n", num_devices);
        	printf("Device: %s (%s)\n", gpu->device_name,value);
	#endif

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	gpu->context = clCreateContext( NULL, 1, &gpu->device_id, NULL, NULL, &ret);
	if ( ret != CL_SUCCESS )
	{
		#ifdef DEBUG
			printf("Error: Create context\n");
		#endif
		goto NO_SUCCESS;
	}


	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL create context:\t\t\t%f sec\n", time);
	#endif
	return(0);

NO_SUCCESS:

	destroy_gpu_context(gpu);	
	return(1);

}


static int create_gpu_program_nonunified(struct gpu_context *gpu, char *func, size_t ALLOC)
{

	cl_int ret;
	#ifdef PROFILE
	struct timeval tv;
        double start,end,time;
	#endif


	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	#if defined(DEBUG) || defined(PROFILE)
		gpu->command_queue = clCreateCommandQueue(gpu->context, gpu->device_id, CL_QUEUE_PROFILING_ENABLE, &ret);
	#else
		gpu->command_queue = clCreateCommandQueue(gpu->context, gpu->device_id, 0 , &ret);
	#endif

	if ( ret != CL_SUCCESS )
	{
		#ifdef DEBUG
			printf("Error: Create command queue\n");
		#endif
		goto NO_SUCCESS;
	} 


	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL create command queue:\t\t%f sec\n", time);
	#endif

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	gpu->hA = malloc(ALLOC);
	if ( gpu->hA == NULL)
	{

		#ifdef DEBUG
			printf("Error: malloc hA\n");
		#endif
		goto NO_SUCCESS;
	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL malloc hA:\t\t\t%f sec\n", time);
	#endif


	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif


	gpu->hB = malloc(ALLOC);
	if ( gpu->hB == NULL)
	{
		#ifdef DEBUG
			printf("Error: malloc hA\n");
		#endif
		goto NO_SUCCESS;
	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL malloc hB:\t\t\t%f sec\n", time);
	#endif


	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	gpu->hC = malloc(ALLOC);
	if ( gpu->hC == NULL)
	{
		#ifdef DEBUG
			printf("Error: malloc hA\n");
		#endif
		goto NO_SUCCESS;
	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL malloc hC:\t\t\t%f sec\n", time);
	#endif


	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif


	gpu->A = clCreateBuffer(gpu->context, CL_MEM_READ_WRITE , ALLOC , NULL, &ret);
	if ( ret != CL_SUCCESS )
	{
		#ifdef DEBUG
			printf("Error: create buffer\n");
		#endif
		goto NO_SUCCESS;
	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL create buffer A:\t\t\t%f sec\n", time);
	#endif


	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	gpu->B = clCreateBuffer(gpu->context, CL_MEM_READ_WRITE , ALLOC , NULL, &ret);
	if ( ret != CL_SUCCESS )
	{
		#ifdef DEBUG
			printf("Error: create buffer\n");
		#endif
		goto NO_SUCCESS;
	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL create buffer B:\t\t\t%f sec\n", time);
	#endif



	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	gpu->C = clCreateBuffer(gpu->context, CL_MEM_READ_WRITE , ALLOC , NULL, &ret);
	if ( ret != CL_SUCCESS )
	{
		#ifdef DEBUG
			printf("Error: create buffer\n");
		#endif
		goto NO_SUCCESS;
	}

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
        	time=end-start;
        	printf("OpenCL create buffer C:\t\t\t%f sec\n", time);
	#endif



	ret = build_gpu_program(gpu, func);
	if ( ret )
	{

		goto NO_SUCCESS;
	}
	

	return(0);

NO_SUCCESS:

	destroy_gpu_context(gpu);	
	return(1);

}



static void open_gpu() {

	#ifdef PROFILE
        struct timeval tv;
        double start,end,time;
	#endif

	cl_int ret;

	#ifdef PROFILE
		printf("----------------------------------------------------------------------------------\n");
		printf("BEGIN Opencl constructor\n");
	#endif

	memset(&gpu, 0, sizeof(struct gpu_context));

	#ifdef PROFILE
		gettimeofday(&tv,NULL);
       		start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

       	ret = create_gpu_context(&gpu);
       	if ( ret )
	{
		have_gpu_context = 0;
              	return;

	}
	have_gpu_context = 1;
	gpu_context = gpu.context;

	#ifdef PROFILE
       		gettimeofday(&tv,NULL);
       		end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
		time=end-start;
		printf("END Opencl constructor\n");
		printf("OpenCL create context summary:\t\t%f sec\n", time);
	#endif

	#ifdef PROFILE
		printf("----------------------------------------------------------------------------------\n");
	#endif

}

static void close_gpu() {

	#ifdef PROFILE
        struct timeval tv;
        double start,end,time;
	#endif

	#ifdef PROFILE
		printf("----------------------------------------------------------------------------------\n");
		printf("BEGIN Opencl destructor\n");
		gettimeofday(&tv,NULL);
       		start=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	destroy_gpu_context(&gpu);
	have_gpu_context = 0;

	#ifdef PROFILE
       		gettimeofday(&tv,NULL);
       		end=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
		time=end-start;
		printf("OpenCL destroy all:\t\t\t%f sec\n", time);
		printf("END Opencl destructor\n");
	#endif


}


static int sgemm_gpu_kernel(struct gpu_context *gpu_ptr, int M, int N, int K, float ALPHA, int acopy, int bcopy , double *ktime)
{

	size_t global_size[3];
        size_t local_size[3];

	#ifdef PROFILE
        struct timeval tv;
        double startg ,endg ,timeg;
	#endif

        cl_int ret = 0;

        global_size[0] = M/SGEMM_GLOBAL0_DIV ;
        global_size[1] = N/SGEMM_GLOBAL1_DIV ;

        local_size[0]  = SGEMM_LOCAL0;
        local_size[1]  = SGEMM_LOCAL1;

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	startg=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

        gpu_ptr->kernel = clCreateKernel(gpu_ptr->program, "sgemm_kernel", &ret);

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	endg=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
		timeg=endg-startg;
		printf("OpenCL create kernel:\t\t\t%f sec\n", timeg);
	#endif


	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	startg=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
	#endif

	ret |= clSetKernelArg(gpu_ptr->kernel, 0, sizeof(int), (void *) &M);
        ret |= clSetKernelArg(gpu_ptr->kernel, 1, sizeof(int), (void *) &N);
        ret |= clSetKernelArg(gpu_ptr->kernel, 2, sizeof(int), (void *) &K);
        ret |= clSetKernelArg(gpu_ptr->kernel, 3, sizeof(float), (void *) &ALPHA);
        ret |= clSetKernelArg(gpu_ptr->kernel, 4, sizeof(cl_mem), (void *) &gpu_ptr->A);
        ret |= clSetKernelArg(gpu_ptr->kernel, 5, sizeof(cl_mem), (void *) &gpu_ptr->B);
        ret |= clSetKernelArg(gpu_ptr->kernel, 6, sizeof(cl_mem), (void *) &gpu_ptr->C);

	#ifdef PROFILE
        	gettimeofday(&tv,NULL);
        	endg=(double) tv.tv_sec+(double)tv.tv_usec*1.e-6;
		timeg=endg-startg;
		printf("OpenCL kernel args:\t\t\t%f sec\n", timeg);
	#endif


	if ( acopy != 0 )
	{

	#ifdef PROFILE
		cl_event perf_event;
        	cl_ulong start, end;
		ret |= clEnqueueWriteBuffer(gpu_ptr->command_queue, gpu_ptr->A, CL_TRUE, 0, M * K *sizeof(float), gpu_ptr->hA, 0, NULL, &perf_event);
        	clWaitForEvents(1, &perf_event);
        	clGetEventProfilingInfo(perf_event,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        	clGetEventProfilingInfo(perf_event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        	printf("Prof: Enqueue A:\t\t\t%f sec\n", (double) (end - start) * (double) ( 1e-9) );
	#else
		ret |= clEnqueueWriteBuffer(gpu_ptr->command_queue, gpu_ptr->A, CL_TRUE, 0, M * K *sizeof(float), gpu_ptr->hA, 0, NULL, NULL);
	#endif

	}

	if ( bcopy != 0 )
	{
	#ifdef PROFILE
		cl_event perf_event;
        	cl_ulong start, end;
		ret |= clEnqueueWriteBuffer(gpu_ptr->command_queue, gpu_ptr->B, CL_TRUE, 0, N * K *sizeof(float), gpu_ptr->hB, 0, NULL, &perf_event);
        	clWaitForEvents(1, &perf_event);
        	clGetEventProfilingInfo(perf_event,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        	clGetEventProfilingInfo(perf_event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        	printf("Prof: Enqueue B:\t\t\t%f sec \n", (double) (end - start) * (double) ( 1e-9) );
	#else
		ret |= clEnqueueWriteBuffer(gpu_ptr->command_queue, gpu_ptr->B, CL_TRUE, 0, N * K *sizeof(float), gpu_ptr->hB, 0, NULL, NULL);
	#endif
	}

	#ifdef PROFILE
		cl_event perf_event;
        	cl_ulong start, end;
        	ret = clEnqueueNDRangeKernel(gpu_ptr->command_queue,gpu_ptr->kernel, 2, NULL, global_size, local_size , 0, NULL, &perf_event);
        	clWaitForEvents(1, &perf_event);
        	clGetEventProfilingInfo(perf_event,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        	clGetEventProfilingInfo(perf_event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        	printf("Prof: Kernel:\t\t\t\t%f sec\n", (double) (end - start) * (double) ( 1e-9) );
		*ktime += (double) (end - start) * (double) ( 1e-9);
	#else
		ret |= clEnqueueNDRangeKernel(gpu_ptr->command_queue, gpu_ptr->kernel, 2, NULL, global_size, local_size , 0, NULL, NULL);
	#endif

	#ifdef PROFILE
		ret |= clEnqueueReadBuffer(gpu_ptr->command_queue, gpu_ptr->C, CL_TRUE, 0, M * N *sizeof(float), gpu_ptr->hC, 0, NULL, &perf_event);
        	clWaitForEvents(1, &perf_event);
        	clGetEventProfilingInfo(perf_event,CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        	clGetEventProfilingInfo(perf_event,CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        	printf("Prof: Enqueue C:\t\t\t%f sec\n", (double) (end - start) * (double) ( 1e-9) );
	#else
		ret |= clEnqueueReadBuffer(gpu_ptr->command_queue, gpu_ptr->C, CL_TRUE, 0, M * N *sizeof(float), gpu_ptr->hC, 0, NULL, NULL);
	#endif

	return(ret);
}


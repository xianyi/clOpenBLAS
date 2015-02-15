#!/bin/bash

if [ -z "$OPENBLAS_DIR" ]; then 

	if [ -f ../libopenblas.so ];then
		OPENBLAS_DIR=`pwd`/..
	elif [ -f ../../libopenblas.so ];then
		OPENBLAS_DIR=`pwd`/../..
	elif [ -f libopenblas.so ];then
		OPENBLAS_DIR=`pwd`
	fi
fi


if [ -z "$OCL_DIR" ]; then
	OCL_DIR=$OPENBLAS_DIR/opencl
fi

# required:
# set the search path for shared libraries
# if not set at an another place
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENBLAS_DIR:$OCL_DIR/lib

# required: search path for opencl source files
if [ -z "$OPENBLAS_CL_DIR" ]; then
	export OPENBLAS_CL_DIR=$OCL_DIR/libcl
fi

# required: the gpu library, that you want to use
if [ -z "$OPENBLAS_GPU_LIB" ]; then
	export OPENBLAS_GPU_LIB=libopenblas_geforce_7xx.so
fi

# optional: export a path to the icd files
# export OPENCL_VENDOR_PATH=$HOME/OpenCL/vendors


# optional: set a search string for a device
# if the device is not found or you want to test the library with another device

# export OPENBLAS_CL_DEVICE=oland
# export OPENBLAS_CL_DEVICE=spectre
# export OPENBLAS_CL_DEVICE=geforce_gts
# export OPENBLAS_CL_DEVICE=geforce_gtx
# export OPENBLAS_CL_DEVICE=geforce
# export OPENBLAS_CL_DEVICE=cpu

# optional: mask available Nvidia devices
# export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0,1

# optional: mask available AMD devices
# export GPU_DEVICE_ORDINAL=0
# export GPU_DEVICE_ORDINAL=0,1

# optional: use mixed precision for dgemm on the gpu
# if you don't need high accuracy
# export OPENBLAS_PRECISION=mixed

# optional: set Minimum sizes for gemm functions or disable the function by setting -1
export SGEMM_GPU_MINSIZE=2048
export DGEMM_GPU_MINSIZE=-1
export CGEMM_GPU_MINSIZE=1024
export ZGEMM_GPU_MINSIZE=-1

# optional: always preload libopenblas_wrap.so
# export LD_PRELOAD=$OCL_DIR/lib/libopenblas_wrap.so:$LD_PRELOAD


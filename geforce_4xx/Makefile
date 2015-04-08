TOPDIR  = ..
include $(TOPDIR)/Makefile.system

# GPU_PFLAGS=-DPROFILE -DDEBUG


All: ${LIBNAME}

test: test-sgemm-${DEVICE_NAME} test-dgemm-${DEVICE_NAME} test-dsgemm-${DEVICE_NAME}

install:
	install ${LIBNAME} ../lib
	install *.cl ../libcl



${DEVICE_NAME}.o: ${DEVICE_NAME}.c ${DEVICE_NAME}.h ../include/gpu_common.h ../gpu_common/sgemm.c ../gpu_common/gemm_common.c ../gpu_common/dgemm.c ../gpu_common/cgemm.c ../gpu_common/zgemm.c ../gpu_common/dsgemm.c
	$(CC) $(CFLAGS) -c $(GPU_PFLAGS) -I../include -I$(TOPDIR)   -o $@ $<


${LIBNAME}: ${DEVICE_NAME}.o 
	$(CC) ${GENLIB_FLAGS} -o $@ $^ ${CLOPENBLAS_EXTRALIB}

test-sgemm-${DEVICE_NAME}: test-sgemm.c ${LIBNAME}
	$(CC) $(CFLAGS) -fopenmp -o test-sgemm-${DEVICE_NAME} test-sgemm.c ${CLOPENBLAS_TESTLIB} $(FEXTRALIB) 


test-dgemm-${DEVICE_NAME}: test-dgemm.c ${LIBNAME}
	$(CC) $(CFLAGS) -fopenmp -o test-dgemm-${DEVICE_NAME} test-dgemm.c ${CLOPENBLAS_TESTLIB} $(FEXTRALIB)


test-cgemm-${DEVICE_NAME}: test-cgemm.c ${LIBNAME}
	$(CC) $(CFLAGS) -fopenmp -o test-cgemm-${DEVICE_NAME} test-cgemm.c ${CLOPENBLAS_TESTLIB} $(FEXTRALIB)


test-zgemm-${DEVICE_NAME}: test-zgemm.c ${LIBNAME}
	$(CC) $(CFLAGS) -fopenmp -o test-zgemm-${DEVICE_NAME} test-zgemm.c ${CLOPENBLAS_TESTLIB} $(FEXTRALIB)

test-dsgemm-${DEVICE_NAME}: test-dsgemm.c ${LIBNAME}
	$(CC) $(CFLAGS) -fopenmp -o test-dsgemm-${DEVICE_NAME} test-dsgemm.c ${CLOPENBLAS_TESTLIB} $(FEXTRALIB)



clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*
	rm -f test-dgemm-*
	rm -f test-dsgemm-*
	rm -f test-cgemm-*
	rm -f test-zgemm-*
	rm -f test-dsgemm-*



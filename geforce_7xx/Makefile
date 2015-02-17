TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

# GPU_PFLAGS=-DPROFILE -DDEBUG

All: libopenblas_geforce_7xx.so

test: test-sgemm-geforce_7xx test-dgemm-geforce_7xx test-dsgemm-geforce_7xx test-cgemm-geforce_7xx

install:
	install libopenblas*.so ../lib
	install *.cl ../libcl



geforce_7xx.o: geforce_7xx.c geforce_7xx.h ../include/gpu_common.h ../gpu_common/sgemm.c ../gpu_common/gemm_common.c ../gpu_common/dgemm.c ../gpu_common/dsgemm.c ../gpu_common/cgemm.c
	$(CC) $(CFLAGS) -fopenmp -fPIC -c $(GPU_PFLAGS) -I../include -I$(TOPDIR)   -o geforce_7xx.o geforce_7xx.c


libopenblas_geforce_7xx.so: geforce_7xx.o 
	$(CC)  -fopenmp -shared -fPIC -o libopenblas_geforce_7xx.so geforce_7xx.o $(OPENCL_LIB_DIR) -lOpenCL

test-sgemm-geforce_7xx: test-sgemm.c libopenblas_geforce_7xx.so $(TOPDIR)/libopenblas.so libopenblas_geforce_7xx.so
	$(CC) $(CFLAGS) -fopenmp -o test-sgemm-geforce_7xx test-sgemm.c $(FEXTRALIB)  -L. -lopenblas_geforce_7xx -L$(TOPDIR) -lopenblas -lOpenCL

test-dgemm-geforce_7xx: test-dgemm.c libopenblas_geforce_7xx.so $(TOPDIR)/libopenblas.so libopenblas_geforce_7xx.so
	$(CC) $(CFLAGS)  -fopenmp -o test-dgemm-geforce_7xx test-dgemm.c $(FEXTRALIB)  -L. -lopenblas_geforce_7xx -L$(TOPDIR) -lopenblas

test-dsgemm-geforce_7xx: test-dsgemm.c libopenblas_geforce_7xx.so $(TOPDIR)/libopenblas.so libopenblas_geforce_7xx.so
	$(CC) $(CFLAGS)  -fopenmp -o test-dsgemm-geforce_7xx test-dsgemm.c $(FEXTRALIB)  -L. -lopenblas_geforce_7xx -L$(TOPDIR) -lopenblas


test-cgemm-geforce_7xx: test-cgemm.c libopenblas_geforce_7xx.so $(TOPDIR)/libopenblas.so libopenblas_geforce_7xx.so
	$(CC) $(CFLAGS)  -fopenmp -o test-cgemm-geforce_7xx test-cgemm.c $(FEXTRALIB)  -L. -lopenblas_geforce_7xx -L$(TOPDIR) -lopenblas





clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*
	rm -f test-cgemm-*
	rm -f test-dgemm-*
	rm -f test-dsgemm-*


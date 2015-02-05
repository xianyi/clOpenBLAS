TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

# GPU_PFLAGS=-DPROFILE -DDEBUG

All: libopenblas_geforce_4xx.so

test: test-sgemm-geforce_4xx

install:
	install libopenblas*.so ../lib
	install *.cl ../libcl



geforce_4xx.o: geforce_4xx.c geforce_4xx.h sgemm.c gemm_common.c
	$(CC) $(CFLAGS) -fopenmp -fPIC -c $(GPU_PFLAGS) -I../include -I$(TOPDIR)   -o geforce_4xx.o geforce_4xx.c


libopenblas_geforce_4xx.so: geforce_4xx.o 
	$(CC) -fopenmp -shared -fPIC -o libopenblas_geforce_4xx.so geforce_4xx.o $(OPENCL_LIB_DIR) -lOpenCL

test-sgemm-geforce_4xx: test-sgemm.c libopenblas_geforce_4xx.so $(TOPDIR)/libopenblas.so libopenblas_geforce_4xx.so
	$(CC) $(CFLAGS) -fopenmp -o test-sgemm-geforce_4xx test-sgemm.c -L. -lopenblas_geforce_4xx -L$(TOPDIR) -lopenblas




clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*

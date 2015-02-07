TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

# GPU_PFLAGS=-DPROFILE -DDEBUG

All: libopenblas_spectre.so

test: test-sgemm-spectre test-dgemm-spectre

install:
	install libopenblas*.so ../lib
	install *.cl ../libcl



spectre.o: spectre.c spectre.h sgemm.c gemm_common.c
	$(CC) $(CFLAGS) -fopenmp -fPIC -c $(GPU_PFLAGS) -I../include -I$(TOPDIR)   -o spectre.o spectre.c


libopenblas_spectre.so: spectre.o 
	$(CC) -fopenmp -shared -fPIC -o libopenblas_spectre.so spectre.o $(OPENCL_LIB_DIR) -lOpenCL

test-sgemm-spectre: test-sgemm.c libopenblas_spectre.so $(TOPDIR)/libopenblas.so libopenblas_spectre.so
	$(CC) $(CFLAGS) -fopenmp -o test-sgemm-spectre test-sgemm.c -L. -lopenblas_spectre -L$(TOPDIR) -lopenblas


test-dgemm-spectre: test-dgemm.c libopenblas_spectre.so $(TOPDIR)/libopenblas.so libopenblas_spectre.so
	$(CC) $(CFLAGS) -fopenmp -o test-dgemm-spectre test-dgemm.c -L. -lopenblas_spectre -L$(TOPDIR) -lopenblas





clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*
	rm -f test-dgemm-*



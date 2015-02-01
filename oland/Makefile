TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

# GPU_PFLAGS=-DPROFILE -DDEBUG

All: libopenblas_oland.so

test: test-sgemm-oland

install:
	install libopenblas*.so ../lib
	install *.cl ../libcl



oland.o: oland.c oland.h sgemm.c gemm_common.c
	$(CC) $(CFLAGS) -fPIC -c $(GPU_PFLAGS) -I../include -I$(TOPDIR)   -o oland.o oland.c


libopenblas_oland.so: oland.o 
	$(CC) -shared -fPIC -o libopenblas_oland.so oland.o $(OPENCL_LIB_DIR) -lOpenCL

test-sgemm-oland: test-sgemm.c libopenblas_oland.so $(TOPDIR)/libopenblas.so libopenblas_oland.so
	$(CC) $(CFLAGS) -o test-sgemm-oland test-sgemm.c -L. -lopenblas_oland -L$(TOPDIR) -lopenblas




clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*

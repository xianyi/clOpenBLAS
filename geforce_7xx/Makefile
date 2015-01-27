TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

# PFLAG=-DPROFILE -DDEBUG
CFLAG=-O2

All: libopenblas_geforce_7xx.so

test: test-sgemm-geforce_7xx

install:
	install libopenblas*.so ../lib
	install *.cl ../libcl



geforce_7xx.o: geforce_7xx.c geforce_7xx.h sgemm.c
	$(CC) $(CFLAGS) -fPIC -c $(PFLAG) -I../include -I$(TOPDIR)   -o geforce_7xx.o geforce_7xx.c


libopenblas_geforce_7xx.so: geforce_7xx.o 
	$(CC) -shared -fPIC -o libopenblas_geforce_7xx.so geforce_7xx.o $(OPENCL_LIB_DIR) -lOpenCL

test-sgemm-geforce_7xx: test-sgemm.c libopenblas_geforce_7xx.so $(TOPDIR)/libopenblas.so libopenblas_geforce_7xx.so
	$(CC) $(CFLAGS) -o test-sgemm-geforce_7xx test-sgemm.c -L. -lopenblas_geforce_7xx -L$(TOPDIR) -lopenblas




clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*

TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

# PFLAG=-DPROFILE -DDEBUG
CFLAG=-O2

All: libopenblas_geforce_4xx.so

geforce_4xx.o: geforce_4xx.c geforce_4xx.h sgemm.c
	$(CC) $(CFLAGS) -fPIC -c $(PFLAG) -I../include -I$(TOPDIR)   -o geforce_4xx.o geforce_4xx.c


libopenblas_geforce_4xx.so: geforce_4xx.o 
	$(CC) -shared -fPIC -o libopenblas_geforce_4xx.so geforce_4xx.o -lOpenCL

test-sgemm-geforce_4xx: test-sgemm.c libopenblas_geforce_4xx.so $(TOPDIR)/libopenblas.so libopenblas_geforce_4xx.so
	$(CC) $(CFLAGS) -o test-sgemm-geforce_4xx test-sgemm.c -L. -lopenblas_geforce_4xx -L$(TOPDIR) -lopenblas




clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*

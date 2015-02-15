TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

# GPU_PFLAGS=-DDEBUG

All: libopenblas_wrap.so

test: test-sgemm-wrap test-sgemm test-sgemm-cblas test-sgemm-cblas-wrap test-dgemm-wrap

install:
	 install libopenblas*.so ../lib


wrap.o: wrap.c sgemm.c dgemm.c cgemm.c zgemm.c
	$(CC) $(CFLAGS) -fPIC -c $(GPU_PFLAGS) -I$(TOPDIR)  -o wrap.o wrap.c

libopenblas_wrap.so: wrap.o
	$(CC) $(CFLAGS) -shared -fPIC -o libopenblas_wrap.so wrap.o -L$(TOPDIR) -lopenblas -ldl -lpthread

test-sgemm-wrap: test-sgemm.c $(TOPDIR)/libopenblas.so libopenblas_wrap.so
	$(CC) $(CFLAGS) -o test-sgemm-wrap test-sgemm.c $(FEXTRALIB) -L. -lopenblas_wrap

test-sgemm: test-sgemm.c $(TOPDIR)/libopenblas.so
	$(CC) $(CFLAGS) -o test-sgemm test-sgemm.c $(FEXTRALIB) -L$(TOPDIR) -lopenblas

test-sgemm-cblas: test-sgemm-cblas.c $(TOPDIR)/libopenblas.so
	$(CC) $(CFLAGS) -o test-sgemm-cblas test-sgemm-cblas.c $(FEXTRALIB) -L$(TOPDIR) -lopenblas

test-sgemm-cblas-wrap: test-sgemm-cblas.c $(TOPDIR)/libopenblas.so libopenblas_wrap.so
	$(CC) $(CFLAGS) -o test-sgemm-cblas-wrap test-sgemm-cblas.c $(FEXTRALIB) -L. -lopenblas_wrap


test-dgemm-wrap: test-dgemm.c $(TOPDIR)/libopenblas.so libopenblas_wrap.so
	$(CC) $(CFLAGS) -o test-dgemm-wrap test-dgemm.c $(FEXTRALIB) -L. -lopenblas_wrap



clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm
	rm -f test-sgemm-wrap
	rm -f test-dgemm-wrap
	rm -f test-sgemm-cblas
	rm -f test-sgemm-cblas-wrap

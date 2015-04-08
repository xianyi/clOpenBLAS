TOPDIR  = ..
include $(TOPDIR)/Makefile.system

# GPU_PFLAGS=-DPROFILE -DDEBUG


All: ${LIBNAME}

test: test-sgemm-device test-sgemm-cpu test-sgemm-cblas-cpu test-sgemm-cblas-device test-dgemm-device

install:
	install ${LIBNAME} ../lib
	install *.cl ../libcl



wrap.o: wrap.c sgemm.c dgemm.c cgemm.c zgemm.c
	$(CC) $(CFLAGS) -c $(GPU_PFLAGS) -I../include -I$(TOPDIR)   -o $@ $< -ldl -lpthread


${LIBNAME}: ${DEVICE_NAME}.o 
	$(CC) ${GENLIB_FLAGS} -o $@ $^ ${CLOPENBLAS_EXTRALIB}

test-sgemm-device: test-sgemm.c ${LIBNAME}
	$(CC) $(CFLAGS) -o $@ $< $(FEXTRALIB) ${CLOPENBLAS_WRAPLIB}

test-sgemm-cpu: test-sgemm.c
	$(CC) $(CFLAGS) -o $@ $< $(FEXTRALIB) ${CPU_BLAS}

test-sgemm-cblas-cpu: test-sgemm-cblas.c
	$(CC) $(CFLAGS) -o $@ $< $(FEXTRALIB) ${CPU_BLAS}

test-sgemm-cblas-device: test-sgemm-cblas.c ${LIBNAME}
	$(CC) $(CFLAGS) -o $@ $< $(FEXTRALIB) ${CLOPENBLAS_WRAPLIB}

test-dgemm-device: test-dgemm.c ${LIBNAME}
	$(CC) $(CFLAGS) -o $@ $< $(FEXTRALIB) ${CLOPENBLAS_WRAPLIB}

clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-cpu
	rm -f test-sgemm-device
	rm -f test-dgemm-device
	rm -f test-sgemm-cblas-*



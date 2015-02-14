TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

#GPU_PFLAGS=-DPROFILE

All: libopenblas_nodev.so

test: test-sgemm-nodev

install:
	install libopenblas*.so ../lib

nodev.o: nodev.c
	$(CC) $(CFLAGS) -fPIC -c $(GPU_PFLAGS) -I$(TOPDIR) -o nodev.o nodev.c

libopenblas_nodev.so: nodev.o
	$(CC) -shared -fPIC -o libopenblas_nodev.so nodev.o

test-sgemm-nodev: test-sgemm.c $(TOPDIR)/libopenblas.so libopenblas_nodev.so
	$(CC)  $(CFLAGS) -o test-sgemm-nodev test-sgemm.c $(FEXTRALIB) -L. -lopenblas_nodev -L$(TOPDIR) -lopenblas


clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*

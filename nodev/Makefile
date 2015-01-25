TOPDIR  = ../..
include $(TOPDIR)/Makefile.system

#PFLAG=-DPROFILE

All: libopenblas_nodev.so

nodev.o: nodev.c
	gcc $(CFLAGS) -fPIC -c $(PFLAG) -I$(TOPDIR) -o nodev.o nodev.c

libopenblas_nodev.so: nodev.o
	gcc -shared -fPIC -o libopenblas_nodev.so nodev.o

test-sgemm-nodev: test-sgemm.c $(TOPDIR)/libopenblas.so libopenblas_nodev.so
	gcc  $(CFLAGS) -o test-sgemm-nodev test-sgemm.c -L. -lopenblas_nodev -L$(TOPDIR) -lopenblas


clean:
	rm -f *.o
	rm -f *.so
	rm -f test-sgemm-*

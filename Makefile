TOPDIR  = ..
include $(TOPDIR)/Makefile.system

OCLDIRS = nodev wrap geforce_4xx

ocl:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE) -C $$d  ; \
        fi; \
        done

clean:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE) -C $$d clean ; \
        fi; \
        done




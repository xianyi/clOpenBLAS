TOPDIR  = ..
include $(TOPDIR)/Makefile.system

OCLDIRS = nodev wrap geforce_4xx geforce_7xx

ocl:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE) -C $$d  ; \
        fi; \
        done
test:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE) -C $$d test ; \
        fi; \
        done

install:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE) -C $$d install ; \
        fi; \
        done

clean:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE) -C $$d clean ; \
        fi; \
        done
	rm -f lib/*.so
	rm -f libcl/*.cl



TOPDIR  = .
include $(TOPDIR)/Makefile.system
MAKE1 = make

OCLDIRS = nodev wrap geforce_4xx geforce_7xx oland spectre

ocl:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE1) -C $$d  ; \
        fi; \
        done
test:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE1) -C $$d test ; \
        fi; \
        done

install:
	-mkdir -p lib/
	-mkdir -p libcl/
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE1) -C $$d install ; \
        fi; \
        done

clean:
	for d in $(OCLDIRS) ; \
        do if test -d $$d; then \
          $(MAKE1) -C $$d clean ; \
        fi; \
        done
	rm -f lib/*.so
	rm -f libcl/*.cl



.PHONY: all build clean coqdoc

all: build

Makefile.coq: _CoqProject
	coq_makefile -f _CoqProject -o Makefile.coq

build: Makefile.coq
	$(MAKE) -f Makefile.coq -j all

clean:
	@if [ -f Makefile.coq ]; then $(MAKE) -f Makefile.coq clean; fi
	rm -f Makefile.coq .coqdeps.d

coqdoc:
	coqdoc --latex -d coqdoc theories/ULCC/*.v

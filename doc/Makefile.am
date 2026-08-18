#---------------------------------------
# doc/Makefile.am for BNCmatmul
# Copyright (c) 2014-2026 Tomonori Kouya
#---------------------------------------

AUTOMAKE_OPTIONS = foreign

# manual.tex     : English  driver  (\JPfalse) -> manual.pdf
# manual_ja.tex  : Japanese driver  (\JPtrue ) -> manual_ja.pdf
# chap01-08.tex  : shared bilingual chapter sources (\bi{English}{Japanese})
EXTRA_DIST = \
	manual.tex manual_ja.tex \
	$(wildcard $(srcdir)/manual.pdf) $(wildcard $(srcdir)/manual_ja.pdf) \
	chap01.tex chap02.tex chap03.tex chap04.tex \
	chap05.tex chap06.tex chap07.tex chap08.tex chap09.tex \
	fig/bncmatmul_fig.pdf \
	fig/lu_strassen_new.pdf \
	fig/multiprec_multifold.pdf \
	fig/ozaki_scheme_3x3.pdf \
	fig/pyrdd.pdf

CHAPTERS = chap01.tex chap02.tex chap03.tex chap04.tex \
	chap05.tex chap06.tex chap07.tex chap08.tex chap09.tex

#---------------------------------------
# 'make pdf' builds both the English (manual.pdf) and Japanese (manual_ja.pdf)
# User's Guides from the same bilingual sources.  The jsbook class needs
# platex + dvipdfmx (pdflatex cannot process it).
#---------------------------------------
pdf-local: manual.pdf manual_ja.pdf

manual.pdf: manual.tex $(CHAPTERS)
	@if command -v platex >/dev/null 2>&1 && command -v dvipdfmx >/dev/null 2>&1; then \
	    cp -f $(srcdir)/*.tex . 2>/dev/null || true ; \
	    test -d fig || cp -R $(srcdir)/fig . 2>/dev/null || true ; \
	    platex -interaction=nonstopmode manual.tex || true ; \
	    platex -interaction=nonstopmode manual.tex || true ; \
	    dvipdfmx manual.dvi || true ; \
	else \
	    echo "platex/dvipdfmx not available - skipping English manual" ; \
	fi

manual_ja.pdf: manual_ja.tex $(CHAPTERS)
	@if command -v platex >/dev/null 2>&1 && command -v dvipdfmx >/dev/null 2>&1; then \
	    cp -f $(srcdir)/*.tex . 2>/dev/null || true ; \
	    test -d fig || cp -R $(srcdir)/fig . 2>/dev/null || true ; \
	    platex -interaction=nonstopmode manual_ja.tex || true ; \
	    platex -interaction=nonstopmode manual_ja.tex || true ; \
	    dvipdfmx manual_ja.dvi || true ; \
	else \
	    echo "platex/dvipdfmx not available - skipping Japanese manual" ; \
	fi

clean-local:
	-rm -f manual.aux manual.log manual.toc manual.out manual.dvi manual.idx \
	       manual_ja.aux manual_ja.log manual_ja.toc manual_ja.out manual_ja.dvi manual_ja.idx

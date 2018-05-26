
all: clean copy latex_start latex_bib latex_end move

latex_start: Report.tex
	pdflatex -output-directory=src/ Report.tex

clean: clean2 #clean1
#clean1: bib.bib
#	rm src/bib.bib

clean2: text_seg.bib
	rm src/text_seg.bib

copy: copy2 #copy1
#copy1: bib.bib
#	cp bib.bib src

copy2: text_seg.bib
	cp text_seg.bib src

latex_bib: latex_bib2 #latex_bib1
#latex_bib1: bib.bib
#	biber src/Report

latex_bib2: text_seg.bib
	biber src/Report

latex_end: latex_end_1 latex_end_2
latex_end_1: Report.tex
	pdflatex -output-directory=src/ Report.tex

latex_end_2: Report.tex
	pdflatex -output-directory=src/ Report.tex

move:
	mv src/Report.pdf ./

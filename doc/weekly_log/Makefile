all: latex1 move

latex1: weekly_log.tex
	pdflatex -output-directory src/ weekly_log.tex

move:
	mv src/weekly_log.pdf ./

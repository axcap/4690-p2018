all: latex1 move

latex1: daily_report.tex
	pdflatex -output-directory src/ daily_report.tex

move:
	mv src/daily_report.pdf ./

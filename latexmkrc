$pdf_mode = 1;
$pdflatex = 'pdflatex -interaction=nonstopmode -halt-on-error %O %S';
$bibtex_use = 2;
$biber = 'biber %O %S';
@default_files = ('main.tex');
$out_dir = 'build';

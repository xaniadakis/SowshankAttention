#!/bin/bash

# compile LaTeX document and its bib references
lualatex -output-directory=out -interaction=nonstopmode Report.tex
biber --input-directory=out Report
lualatex -output-directory=out -interaction=nonstopmode Report.tex

# if production mode, output final PDF in its place
if [ "$1" == "prod" ]; then
    lualatex -output-directory=out -interaction=nonstopmode Report.tex
    cp out/Report.pdf ..
    echo "Production PDF ready."
fi

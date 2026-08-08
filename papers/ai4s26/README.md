# AI4S 2026 Paper Package

`main.tex` is the 5--8 page IEEE-format archival manuscript. It is tailored to the AI4S/SC26 emphasis on scientific surrogates, trustworthy validation, and performance.

Regenerate the paper figures from the corrected result artifacts with:

```bash
.venv/bin/python papers/ai4s26/make_figures.py
```

Compile in an IEEEtran-capable LaTeX environment with:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Tectonic is now installed locally, and `main.pdf` has been compiled and visually inspected. `claims.md` records the evidence and qualification behind each headline claim.

Before submission, replace the anonymous author block as required by AI4S's single-blind review, add the artifact URL and license, and substantively revise the AI-assisted draft in the authors' own voice.

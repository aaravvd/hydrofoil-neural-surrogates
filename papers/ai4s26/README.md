# AI4S 2026 Paper Package

`main.tex` contains approximately seven pages of IEEE-format main text; the
bibliography begins near the end of page seven and continues on page eight.
References do not count toward the AI4S limit. The manuscript is tailored to
the AI4S/SC26 emphasis on scientific surrogates, trustworthy validation, and
performance.

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

The author block, public GitHub URL, Zenodo DOI, MIT license, and required AI-tool disclosure are included. AI4S uses single-blind review, so the paper and artifact do not need to be anonymized. Before upload, confirm the author affiliation in `main.tex`, create an immutable GitHub submission tag, and run one final citation and PDF check.

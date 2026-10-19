# Sim2Science 2026 Camera-Ready Paper

This directory contains the accepted camera-ready paper using the unmodified
official NeurIPS 2026 style and the required workshop configuration:

```latex
\usepackage[dblblindworkshop, final]{neurips_2026}
\workshoptitle{Sim2Science: ML with Imperfect Scientific Models}
```

Compile from this directory with:

```bash
TECTONIC_CACHE_DIR="../../.cache/tectonic" tectonic main.tex
```

The technical paper occupies five pages. References, appendices, and the
completed official NeurIPS checklist follow outside that limit.

# Wildfire — agent instructions

## Logging

**ALWAYS log experiments and research in `blog/posts/`.**

One directory per entry, `blog/posts/YYYY-MM-DD-slug/index.qmd`. Start from
`blog/posts/_template/index.qmd`, which fixes the required front matter
(`title`, `description`, `date`, `categories`, `verdict`) and the sections
every entry carries: Question, Setup, Result, Verdict, Evidence.

`verdict` is `accepted`, `rejected`, or `open`. Rejected results are logged with
the same care as accepted ones — they are what stops the next agent repeating
the work.

Before proposing an experiment, read `blog/experiments.qmd` output or the
existing posts. Do not re-run an experiment that already has a verdict unless
you have new evidence, and if you do, say in the post what changed.

Render locally with `quarto render blog` before pushing; CI publishes to
GitHub Pages on push to `main`.

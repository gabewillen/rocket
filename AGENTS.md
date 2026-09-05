# Wildfire — agent instructions

## Choosing what to build

Wildfire exists to make the impossible possible. Assume the easy roads have
already been traveled, and that the baselines recorded under "Prior art" in
`blog/design.qmd` are exactly what they yield.

- ALWAYS assume that any easy, obvious, or well-documented approach has already been tried by a capable team, and that its result is already recorded as a baseline.
- NEVER propose work whose best case is matching what a general-purpose engine already achieves; if the upside is bounded by a published baseline, it is not worth doing here.
- ALWAYS state, for any approach you propose, why it has not already been done — genuine difficulty, hardware this new, or specificity to this one model. If there is no such reason, assume it has been done and go find its result.
- NEVER treat "unsupported", "not in the support matrix", "no wheel published", or "not in the model zoo" as a stopping condition; ALWAYS convert it into a build flag, a port, a plugin, or a measurement.
- NEVER reject an approach because it has not been done before, and NEVER accept one because it is conventional.
- ALWAYS separate difficulty objections from value objections when rejecting something, and ALWAYS say which kind you are making — only value objections survive someone willing to do the work.
- ALWAYS prefer the hard path when the easy path's ceiling is already known and is not enough.

## Build log

The log exists so no experiment is run twice without new evidence. Every rule
below serves that, and every rule is stated as ALWAYS or NEVER because a rule
you can argue with is a rule that gets argued with.

### It is public

The repository and the rendered log are public. Everything below assumes a
reader outside the project, and assumes git history is as public as the working
tree.

- ALWAYS write entries for a public audience, and NEVER write anything you would not want read by the maintainers of the projects being measured.
- NEVER commit a credential of any kind — tokens, keys, cookies, `.env` files, `hosts.yml` — and NEVER paste command output that embeds one.
- ALWAYS scan `git diff --cached` before committing evidence files, because raw logs and JSON are where a token actually hides.
- NEVER remediate a leaked secret by deleting it in a later commit; ALWAYS rotate it first, then purge history, because the earlier commit stays public until you do.
- NEVER publish material from private, embargoed, or NDA-covered sources, including internal benchmarks belonging to someone else.
- NEVER publish personal information, including names, emails, or account handles other than your own project identity.
- ALWAYS report another project's numbers with its version, commit, and configuration, so the comparison is reproducible and fair to it.
- NEVER characterize another project as bad; ALWAYS report the measurement and the configuration and let the reader judge.

### Where it goes

- ALWAYS log experiments and research in `blog/posts/`.
- ALWAYS create one directory per entry at `blog/posts/YYYY-MM-DD-slug/index.qmd`, so evidence files live beside the prose that cites them.
- ALWAYS treat that path as the entry's permanent identifier; it is what commits, spec notes, and other posts cite.
- NEVER reuse a slug, and NEVER renumber, rename, or move a published entry.
- ALWAYS start from `blog/posts/_template/index.qmd` rather than composing front matter by hand.

### How it is indexed

- ALWAYS set every required front matter field: `title`, `description`, `date`, `author`, `categories`, `verdict`.
- ALWAYS write `title` as the claim, not the topic: "NVFP4 grouped GEMM beats EXL3 at batch 8", never "GEMM investigation".
- ALWAYS write `description` as one sentence that states the outcome, because it is the only text a search result shows and it must be enough to decide whether to open the post.
- ALWAYS put the kind first in `categories`, from exactly one of `experiment`, `research`, `incident`, `infrastructure`.
- ALWAYS follow the kind with every affected component, drawn only from `compiler`, `runtime`, `memory`, `kernels`, `attention`, `moe`, `cache`, `fabric`, `scheduler`, `numerics`, `telemetry`, `tooling`.
- NEVER invent a category outside those two vocabularies; extending a vocabulary is an edit to this file first.
- ALWAYS set `verdict` to exactly one of `accepted`, `rejected`, `open`.
- NEVER leave `verdict: open` once the measurement has run — an unresolved verdict is an invitation to repeat the work.
- ALWAYS add `design:` with the anchor on `blog/design.qmd` the result affects, when it affects one.
- ALWAYS keep `blog/` the single home for project information: current state on the Design page, history in the log. NEVER add a standalone design or status document at the repository root.

### What an entry must contain

- ALWAYS record, under Setup, the node the work ran on, the exact commit SHA, and the literal copy-pasteable command.
- ALWAYS report numbers with units in a table, and NEVER report a measurement in prose alone.
- ALWAYS state run-to-run variance when a single number would be misleading.
- ALWAYS commit evidence as JSON in the entry's `evidence/` directory and link it from the Evidence section, so results are machine-readable and comparable across entries.
- ALWAYS give every evidence file at minimum `generated_utc`, the `question` it answers, the `answer`, and the `sources` (URL, command, or file) each value came from.
- NEVER commit an unstructured console dump as the evidence of record; a `.txt` transcript is neither renderable nor comparable. Attach large raw logs alongside the JSON only when the JSON references them.
- ALWAYS state, in the Verdict section, what would have to change for the question to be reopened.
- ALWAYS log a rejected result with the same care as an accepted one; rejections are what stop the next agent from repeating the work.

### Before and after

- ALWAYS search the existing log before proposing an experiment, with `rg` over `blog/posts/` or the site search at <https://gabewillen.github.io/wildfire/>.
- NEVER re-run an experiment that already carries a verdict unless you have new evidence.
- ALWAYS add `supersedes:` pointing at the earlier entry's path when you do re-run one, and ALWAYS say in the post what changed.
- NEVER edit a published entry's findings to reflect a later result; supersede it instead, because the log is append-only history and not current state.
- ALWAYS run `quarto render blog` locally before pushing, since CI publishes to GitHub Pages on every push to `main`.

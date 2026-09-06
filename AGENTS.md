# Rocket: agent instructions

## Choosing what to build

Assume the easy roads have already been traveled, and that the recorded
baselines are what they yield.

- ALWAYS assume that any easy, obvious, or well-documented approach has already been tried by a capable team, and that its result is already recorded as a baseline.
- NEVER propose work whose best case is matching what a general-purpose engine already achieves; if the upside is bounded by a published baseline, it is not worth doing here.
- ALWAYS state why a proposed approach has not already been done: difficulty, hardware this new, or specificity to one model. If there is no such reason, assume it has been done and go find its result.
- NEVER treat "unsupported", "not in the support matrix", "no wheel published", or "not in the model zoo" as a stopping condition; ALWAYS convert it into a build flag, a port, a plugin, or a measurement.
- NEVER reject an approach because it has not been done before, and NEVER accept one because it is conventional.
- ALWAYS say whether a rejection is a difficulty objection or a value objection. Difficulty objections do not hold against someone willing to do the work.
- ALWAYS prefer the hard path when the easy path's ceiling is already known and is not enough.
- NEVER optimize without a profile naming the bottleneck first; a counter, a stall reason, or a roofline gap, quoted in the entry.
- ALWAYS size the change to the iteration cost: when one attempt costs minutes of model load, make the large structural change first and bisect from it. Parameter tweaks come after the structure wins, as controls.

## Build log

The log exists so no experiment is run twice without new evidence.

### It is public

The repository and the rendered log are public. Everything below assumes a
reader outside the project, and assumes git history is as public as the working
tree.

- ALWAYS write entries for a public audience, and NEVER write anything you would not want read by the maintainers of the projects being measured.
- NEVER commit a credential of any kind (tokens, keys, cookies, `.env` files, `hosts.yml`), and NEVER paste command output that embeds one.
- ALWAYS scan `git diff --cached` before committing, because pasted command output is where a token actually hides.
- NEVER remediate a leaked secret by deleting it in a later commit; ALWAYS rotate it first, then purge history, because the earlier commit stays public until you do.
- NEVER publish material from private, embargoed, or NDA-covered sources, including internal benchmarks belonging to someone else.
- NEVER publish personal information, including names, emails, or account handles other than your own project identity.
- ALWAYS report another project's numbers with its version, commit, and configuration, so the comparison is reproducible and fair to it.
- NEVER characterize another project as bad; ALWAYS report the measurement and the configuration and let the reader judge.

### Where it goes

- ALWAYS log experiments and research in `blog/posts/`.
- ALWAYS create one directory per entry at `blog/posts/<area>/YYYY-MM-DD-slug/index.qmd`, where `<area>` is the primary component. Subpaths keep `rg blog/posts/hardware/` cheap without parsing front matter.
- ALWAYS treat that path as the entry's permanent identifier; it is what commits and other entries cite.
- NEVER reuse a slug, and NEVER renumber, rename, or move a published entry.
- ALWAYS start from `blog/posts/_template/index.qmd` rather than composing front matter by hand.

### How it is indexed

- ALWAYS set every required front matter field: `title`, `description`, `date`, `author`, `categories`, `verdict`.
- ALWAYS write `title` as the claim, not the topic: "NVFP4 grouped GEMM beats EXL3 at batch 8", never "GEMM investigation".
- ALWAYS write `description` as one sentence that states the outcome, because it is the only text a search result shows and it must be enough to decide whether to open the post.
- ALWAYS put the kind first in `categories`, from exactly one of `experiment`, `research`, `incident`, `infrastructure`.
- ALWAYS follow the kind with every affected component, drawn only from `hardware`, `compiler`, `runtime`, `memory`, `kernels`, `attention`, `moe`, `cache`, `fabric`, `scheduler`, `numerics`, `telemetry`, `tooling`, `baseline`.
- ALWAYS use the first component as `<area>` in the entry path.
- NEVER invent a category outside those two vocabularies; extending a vocabulary is an edit to this file first.
- ALWAYS set `verdict` to exactly one of `accepted`, `rejected`, `open`.
- NEVER leave `verdict: open` once the measurement has run.
- ALWAYS add `rocket:` with the anchor on `blog/rocket.qmd` the result affects, when it affects one.
- ALWAYS keep `blog/` the single home for project information: current state on the Rocket page, history in the log. NEVER add a standalone design or status document at the repository root.

### What an entry must contain

- ALWAYS use the `/self-voice` skill to write entries, and be short and concise, avoiding unnecessary prose.

### Do not write like a chatbot

- NEVER use em dashes. Use a period, a comma, or parentheses.
- NEVER open a section with a bold one-line pronouncement.
- NEVER use "not X, but Y" or "X is not Y, it is Z" constructions.
- NEVER end a section with an aphorism, a moral, or a summary of what the reader just read.
- NEVER stack three adjectives or three clauses for rhythm.
- NEVER use these words: genuinely, precisely, deliberately, honest, crucial, robust, seamless, comprehensive, load-bearing, worth noting, that said, the whole point, the key insight.
- NEVER explain the significance of a fact you just stated. State it and stop.
- ALWAYS prefer a number to an adjective.
- ALWAYS cut any sentence that does not carry a fact, a decision, or an instruction.
- NEVER write narrative or explanatory paragraphs; ALWAYS prefer a table, a one-line bullet, or a single sentence.
- NEVER restate in prose what the front matter or a table already says.
- ALWAYS keep an entry scannable in under a minute; if it needs more room, the detail belongs on the Rocket page.

- ALWAYS record, under Setup, the node the work ran on, the exact commit SHA, and the literal copy-pasteable command.
- ALWAYS report numbers with units in a table, and NEVER report a measurement in prose alone.
- ALWAYS state run-to-run variance when a single number would be misleading.
- ALWAYS make the entry itself the evidence. NEVER add an `evidence/` directory or attach `.json`, `.txt`, or `.log` files.
- ALWAYS show the command that produced a number, and its output, inline in the `.qmd` as a fenced block or a table.
- ALWAYS link the source for a number taken from outside this project, so a reader can check it.
- NEVER write a script to `/tmp` or any other scratch path. ALWAYS commit every script that produced a number, under `scripts/<area>/`, and cite it by repository path in the entry.
- ALWAYS make a committed script runnable as-is, so a reader reproduces the number without reconstructing it from a fenced block.
- ALWAYS state, in the Verdict section, what would have to change for the question to be reopened.
- NEVER put work this project can do itself in a reopen condition; reopen conditions are for external change only (firmware, upstream releases, hardware). Doable work goes in a "Next" list and gets done.
- ALWAYS log a rejected result with the same care as an accepted one.

### Before and after

- ALWAYS search the existing log before proposing an experiment, with `rg` over `blog/posts/` or the site search at <https://gabewillen.github.io/rocket/>.
- NEVER re-run an experiment that already carries a verdict unless you have new evidence.
- ALWAYS add `supersedes:` pointing at the earlier entry's path when you do re-run one, and ALWAYS say in the post what changed.
- NEVER edit a published entry's findings to reflect a later result; supersede it instead, because the log is append-only history and not current state.
- ALWAYS run `quarto render blog` locally before pushing, since CI publishes to GitHub Pages on every push to `main`.

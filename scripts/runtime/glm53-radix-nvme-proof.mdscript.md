<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Prove State Invariance

* inherit `{{repo}}`, `{{engine}}`, `{{fuel}}`, `{{cache_dir}}`, `{{peer}}`, `{{staging_bytes}}`, and `{{objective}}` from the caller
* add a focused test that runs one token prefix at M1, M8, and M16
* compare target MLA latent bytes at every cached layer
* compare indexer key and gate bytes at every cached layer
* compare FP32 KDA recurrent and convolution state
* compare DFlash2 restorable state
* if independently computed bytes differ by execution shape
  * namespace cache keys by execution shape
* if independently computed bytes match at all tested shapes
  * permit deduplication under the common numerical namespace
* never weaken the test to token-only parity
* [Test Lifecycle And Recovery](#test-lifecycle-and-recovery)

## Test Lifecycle And Recovery

* test two requests sharing one full prefix page
* test two requests branching after a shared prefix
* test copy-on-extend for a shared partial page
* test coalesced restore into two stream page tables
* test detach, eviction, restore, resume, and append
* test missing target component fallback
* test missing KDA checkpoint fallback
* test missing DFlash2 component fallback
* test checksum corruption fallback
* test truncated record recovery
* test manifest replay after simulated process restart
* test capacity eviction preserves protected ancestors
* test rank-boundary disagreement falls back to the common ancestor
* test configuration parsing for `512GiB`
* test startup rejection when physical free-space headroom is insufficient
* test pinned staging never exceeds `{{staging_bytes}}`
* if any lifecycle test fails
  * fix the cause and [Test Lifecycle And Recovery](#test-lifecycle-and-recovery)
* [Measure Crossover](#measure-crossover)

## Measure Crossover

* run the two-request 2k shared-prefix trace with restore enabled
* run the two-request 8k shared-prefix trace with restore enabled
* run the two-request 32k shared-prefix trace with restore enabled
* compare restore time against full recompute time at each length
* report NVMe read and write throughput with units
* report bytes restored per reused token
* report time to first new token after admission
* report run-to-run variance from at least three restore replicates at the winning length
* reject NVMe restore if it fails to beat recompute at every tested length
* if NVMe restore is rejected
  * preserve in-memory radix sharing and remove the losing NVMe production default
* [Run Decode Non Regression](#run-decode-non-regression)

## Run Decode Non Regression

* run the committed c8 256-token decode gate with cache disabled
* run the same gate with cache enabled but no hit
* run the same gate after a restored hit
* require generated token IDs to match the cache-disabled reference
* require draft acceptance to remain within one percentage point
* require decode throughput after admission to remain within three percent
* run c16 at its measured best draft depth of four
* require no new fabric timeout or memory-pressure regression
* if a decode gate fails
  * profile the first divergent state component and [Wire Radix Admission](glm53-radix-nvme-goal.mdscript.md#wire-radix-admission)
* [Run Full Proof](#run-full-proof)

## Run Full Proof

* build every target in `{{engine}}/build`
* run the KV radix tests
* run the real KV fork test
* run batch parity last
* run the refreshed two-rank static-fire gate with identical production flags on both ranks
* run `git diff --check`
* scan staged changes for credentials and private data
* if any proof fails
  * fix the cause and [Run Full Proof](#run-full-proof)
* [Publish Evidence](#publish-evidence)

## Publish Evidence

* create one post from `{{repo}}/blog/posts/_template/index.qmd`
* place the post under `blog/posts/cache/YYYY-MM-DD-<unused-slug>/index.qmd`
* state the measured claim in the title
* include node, exact commit SHA, and literal commands
* include baseline, restore, recompute, memory, and decode tables with units
* include the SGLang source links inspected before implementation
* classify rejections as value or difficulty objections
* update `{{repo}}/blog/rocket.qmd`
* run `quarto render blog`
* if rendering fails
  * fix the post and [Publish Evidence](#publish-evidence)
* [Commit And Review](#commit-and-review)

## Commit And Review

* commit implementation and tests atomically
* commit evidence and Rocket status separately
* run multi-lane review for rules, security, completeness, C++, CUDA, storage crash consistency, and state-machine correctness
* require a Proven-for verdict with no blocking findings
* if review finds a blocker
  * fix every blocker and [Run Full Proof](#run-full-proof)
* [Reduce Decode Bytes](glm53-radix-nvme-goal.mdscript.md#reduce-decode-bytes)

## Complete

* verify in-memory sharing works across requests and streams
* verify NVMe restore uses bounded direct IO without page-cache or unified-memory growth
* verify target MLA, FP32 KDA, and DFlash2 resume from one common exact boundary
* verify the 512 GiB capacity is configurable and safely rejected when unavailable
* verify the crossover and decode non-regression evidence is published
* report commits, tests, crossover length, restore bandwidth, memory bound, and remaining limitations
* mark the Pi goal complete only after every verification above has authoritative evidence

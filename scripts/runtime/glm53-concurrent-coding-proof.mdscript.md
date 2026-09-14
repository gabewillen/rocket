<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Run Final Proof

* inherit `{{repo}}`, `{{engine}}`, `{{workload}}`, `{{baseline}}`, and `{{objective}}` from the caller
* verify the workload manifest uses immutable public revisions
* verify every c8 stream has a distinct prompt and tool history
* verify the production run generates at least 4096 useful output tokens
* run the final c8 workload five times
* run the final c4 workload three times
* run the final c16 workload three times
* run the no-speculation c8 control once
* run the perfect-draft ceiling once
* label the perfect-draft result as a ceiling
* record median and run-to-run variance
* [Verify Quality And Causality](#verify-quality-and-causality)

## Verify Quality And Causality

* run the committed teacher-forced corpus at c8 K1
* run the committed teacher-forced corpus at the production verification width
* run the future-token causality test
* run deterministic coding-task checks
* run patch compilation and tests in isolated public fixtures
* require zero future-token influence on earlier logits
* require zero nonfinite logits
* require no coding-task class to regress by more than one pass-rate point
* require perplexity to remain within one percent of `{{baseline}}`
* require mean KL divergence to remain within `0.01` nat per token of `{{baseline}}`
* if any quality or causality gate fails
  * return to [Profile Real Workload](glm53-concurrent-coding-goal.mdscript.md#profile-real-workload)
* [Verify Serving Correctness](#verify-serving-correctness)

## Verify Serving Correctness

* build every target in `{{engine}}/build`
* run kernel tests
* run DFlash2 tests
* run batch-parity tests
* run KV fork tests
* run radix and NVMe prefix-cache tests
* run MoE grouped-invariance tests
* run the two-rank static-fire test with production flags
* run one prefix-cache cold request
* run one exact prefix restore
* require cold and restored generated token IDs to match
* require no fabric timeout
* require no expert-cache miss after owned-expert preload
* require resident memory to leave the recorded host safety headroom
* if any serving gate fails
  * return to [Profile Real Workload](glm53-concurrent-coding-goal.mdscript.md#profile-real-workload)
* [Verify Power And Latency](#verify-power-and-latency)

## Verify Power And Latency

* sample both GB10 GPUs at 200 ms during every final c8 production run
* report aggregate useful output tokens per pair-GPU watt
* report aggregate useful output tokens per pair-GPU joule
* state that NVML excludes CPU, DRAM, SSD, fabric, PSU, and wall losses
* report p50 and p95 time to first token
* report p50 and p95 inter-token latency
* report p50 and p95 completion latency
* report p10 and mean acceptance by task class
* require the final tokens per joule to equal or exceed `{{baseline}}`
* require p95 completion latency to stay within five percent of `{{baseline}}`
* if either efficiency or latency gate fails
  * return to [Profile Real Workload](glm53-concurrent-coding-goal.mdscript.md#profile-real-workload)
* [Publish Evidence](#publish-evidence)

## Publish Evidence

* create a post from `{{repo}}/blog/posts/_template/index.qmd`
* place it under `{{repo}}/blog/posts/runtime/YYYY-MM-DD-<unused-slug>/index.qmd`
* state the measured coding-workload claim in the title
* include the exact workload revision and checksum
* include node, kernel, exact commit SHA, and literal commands
* include c4, c8, and c16 throughput tables
* include tokens per watt and tokens per joule
* include latency distributions
* include acceptance distributions by coding-task class
* include accuracy and coding-task pass-rate deltas
* include prefix-cache and expert-cache counters
* distinguish production measurements from ceiling controls
* log every rejected iteration with its bottleneck and rejection reason
* update `{{repo}}/blog/rocket.qmd`
* run `quarto render blog`
* if rendering fails
  * fix the publication and [Publish Evidence](#publish-evidence)
* [Review And Complete](#review-and-complete)

## Review And Complete

* commit implementation and tests atomically
* commit workload and evidence separately
* run `git diff --check`
* scan staged changes for credentials and private data
* run multi-lane review for rules, security, completeness, C++, CUDA, scheduler behavior, state-machine correctness, numerics, and benchmark validity
* require a Proven-for verdict with no blocking findings
* if review finds a blocker
  * fix every blocker and [Run Final Proof](#run-final-proof)
* report final c4, c8, and c16 useful throughput
* report final c8 tokens per pair-GPU watt
* report coding-task pass rate and numerical deltas
* report remaining hardware and value limits
* mark a harness goal complete only when its active Goal contract supplies the current goal ID and every proof above passes
* otherwise stop and report the proof without calling goal-completion tools

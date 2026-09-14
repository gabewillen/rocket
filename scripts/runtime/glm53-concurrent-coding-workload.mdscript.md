<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Build Coding Workload

* inherit `{{repo}}`, `{{engine}}`, `{{peer}}`, `{{workload}}`, and `{{objective}}` from the caller
* create a committed workload builder under `{{repo}}/scripts/runtime`
* source only public repositories, public issues, or sanitized project-owned fixtures
* pin every external repository and dataset to an immutable revision
* record source URLs and checksums in the workload manifest
* create at least sixteen distinct sessions
* include repository comprehension sessions
* include bug diagnosis sessions
* include patch-generation sessions
* include test-repair sessions
* include code-review sessions
* include refactoring sessions
* include Python, C++, CUDA, Rust, JavaScript, and shell when public fixtures permit
* include prompt contexts near 2k, 8k, 32k, and 64k tokens
* include response budgets of 128, 256, 512, and 1024 tokens
* include at least three turns per session
* include deterministic tool-result injections between turns
* include shared system and repository prefixes for NVMe cache measurement
* preserve distinct user tasks and distinct tool histories per stream
* reject the workload if any c8 prompt hash is duplicated
* reject the workload if generated-token budgets total fewer than 4096 tokens at c8
* [Implement Workload Runner](#implement-workload-runner)

## Implement Workload Runner

* create a separate `rocket-coding-bench` executable or a separate harness around a stable serving API
* keep workload parsing out of `rocket-decode`
* admit independent prompts into independent stream slots
* support staggered arrivals
* support completed-session replacement without resetting active sessions
* append deterministic tool outputs between turns
* use the production NVMe prefix cache
* record per-session generated tokens
* record per-session wall time
* record time to first token
* record inter-token latency
* record completion latency
* record DFlash2 accepted length by draft position
* record target verification time
* record draft proposal time
* record expert-cache hits and misses
* record prefix-cache restored tokens and restore latency
* record routing entropy
* record both-node GPU power samples
* emit one machine-readable result file
* add a test that fails when prompts are replicated
* add a test that fails when token accounting includes padding or rejected drafts
* add a causality test that changes future tokens without changing earlier logits
* if any runner test fails
  * fix the runner and [Implement Workload Runner](#implement-workload-runner)
* [Capture Production Baseline](glm53-concurrent-coding-goal.mdscript.md#capture-production-baseline)

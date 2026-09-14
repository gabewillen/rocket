---
name: glm53-concurrent-coding-goal
description: Recursively profiles and optimizes the GLM-5.3 CUDA engine for sustained, diverse, concurrent coding-agent sessions with quality and energy gates.
---

<!-- mdscript: use the mdscript-exec skill or read [spec.md](https://raw.githubusercontent.com/gabewillen/mdscript/main/spec.md) -->

## Goal Contract

* set `{{repo}}` to `/home/glwillen/Development/rocket`
* set `{{engine}}` to `{{repo}}/engines/glm5-moe-nvfp4-2b`
* set `{{peer}}` to `192.168.100.11`
* set `{{workload}}` to `{{repo}}/scripts/runtime/fixtures/glm53-coding-sessions.jsonl`
* set `{{run_note}}` to `/tmp/glm53-concurrent-coding-run.md`
* set `{{concurrency_list}}` to `4 8 16`
* set `{{primary_concurrency}}` to `8`
* set `{{max_iterations}}` to `10`
* set `{{iteration}}` to `0`
* set `{{minimum_win}}` to `3` percent
* set `{{objective}}` to `maximize sustained useful output throughput and tokens per GPU watt for real concurrent coding sessions without reducing task quality, numerical fidelity, cache correctness, or host stability`
* require eight distinct active coding sessions at c8
* require sixteen distinct active coding sessions at c16
* forbid replicated prompts in production measurements
* forbid counting prompts, repeated-token prompts, forced acceptance, oracle drafts, and synthetic perfect drafts in production measurements
* allow those controls only in separately labeled ceiling runs
* require every production run to use the production fuel, DFlash2, expert split, and NVMe prefix cache defaults
* require one GPU workload at a time across both nodes
* require every optimization proposal to name a measured bottleneck before code changes
* whenever this workflow is read or resumed, report the current active workload
* whenever this workflow is read or resumed, report the latest authoritative c8 and c16 useful throughput
* whenever this workflow is read or resumed, report the current measured bottleneck
* whenever this workflow is read or resumed, report candidates accepted or rejected since the preceding update
* whenever this workflow is read or resumed, report the next measurement or proof gate
* require every accepted result to use committed scripts and an exact commit SHA
* require rejected fuel additives to be removed from the loader and both nodes
* [Inspect Current Evidence](#inspect-current-evidence)

## Inspect Current Evidence

* read `{{repo}}/AGENTS.md`
* read `{{repo}}/scripts/runtime/glm53-nvfp4-spec-bench.sh`
* read `{{repo}}/scripts/runtime/glm53-teacher-score-pair.sh`
* read `{{repo}}/scripts/hardware/glm53-power-bench.sh`
* read the current DFlash2 proposal, verification, commit, and rollback paths
* read the current expert-cache and NVMe prefix-cache paths
* inspect one measured general-purpose engine implementation for every serving primitive considered
* search `{{repo}}/blog/posts` for prior DFlash2, batching, power, cache, and quantization verdicts
* record the favorable-prompt ceiling of `213.91 aggregate tok/s at c8 with 100 percent acceptance`
* record the long counting-to-prose result of `93.20 aggregate tok/s at c8 with 50.9 percent acceptance and 0.808 aggregate tok/s per GPU watt`
* classify the old counting benchmark as a ceiling control
* [Build Coding Workload](glm53-concurrent-coding-workload.mdscript.md#build-coding-workload)

## Capture Production Baseline

* build every engine target
* run the focused correctness tests
* run the c8 workload three times with production defaults
* run one c8 control with speculation disabled
* run one c8 sweep across draft depths one through eight
* run c4 and c16 only for the best two c8 schedules
* record aggregate useful output tok/s
* record tok/s per active stream
* record pair-GPU tokens per joule
* record p50 and p95 time to first token
* record p50 and p95 inter-token latency
* record p50 and p95 completion latency
* record mean and p10 DFlash2 acceptance
* record acceptance by draft position
* record metrics by coding-task class
* record run-to-run variance
* select the median c8 production run as `{{baseline}}`
* if any stream fails to remain active for its declared session duration
  * fix admission or replacement and [Capture Production Baseline](#capture-production-baseline)
* [Run Accuracy Baseline](#run-accuracy-baseline)

## Run Accuracy Baseline

* run `rocket-teacher-score` at c8 K1
* run `rocket-teacher-score` at c8 production verification width
* record perplexity, KL divergence, centered logit RMS, top-1 agreement, and top-5 overlap
* run deterministic coding-task checks for compilable patches, tests, and requested output format
* record pass rate by task class
* treat the same production fuel and numerical shape as the control for kernel-only changes
* require a source-BF16 or NVIDIA reference at the same shape before claiming checkpoint-level accuracy
* if a candidate introduces nonfinite logits or future-token leakage
  * reject it and [Revert Candidate](#revert-candidate)
* [Profile Real Workload](#profile-real-workload)

## Profile Real Workload

* profile the median c8 baseline run
* attribute wall time to draft, KDA, MLA/indexer, MoE, LM head, fabric, cache restore, scheduler, and idle gaps
* measure bytes moved for the largest stage
* measure achieved bandwidth against the GB10 pattern ceiling
* measure SM occupancy and launch gaps for the largest stage
* measure expert-cache locality under distinct sessions
* measure acceptance loss by task class and response position
* set `{{bottleneck}}` to the largest exposed cost in the real workload
* set `{{hypothesis}}` to one structural change that attacks `{{bottleneck}}`
* if no counter, stall reason, or roofline gap supports `{{hypothesis}}`
  * gather the missing profile and [Profile Real Workload](#profile-real-workload)
* [Implement One Structural Change](#implement-one-structural-change)

## Implement One Structural Change

* inspect the corresponding implementation in one measured general-purpose engine
* state why the proposed specialization was not already captured by the baseline
* classify objections as difficulty or value objections
* implement one structural change
* keep source weight dtype separate from serving weight dtype
* keep every replacement fuel additive mutually exclusive with its source weights
* add focused tests for the changed primitive
* rebuild the engine
* if the build or focused tests fail
  * fix the cause and [Implement One Structural Change](#implement-one-structural-change)
* [Measure Candidate](#measure-candidate)

## Measure Candidate

* run the unchanged c8 workload three times
* run the unchanged accuracy baseline
* run the unchanged power measurement
* compare medians against `{{baseline}}`
* compare p95 latency against `{{baseline}}`
* compare each task class against `{{baseline}}`
* compare DFlash2 acceptance distributions against `{{baseline}}`
* compare cache and expert locality against `{{baseline}}`
* accept only useful output tokens in throughput accounting
* [Decide Candidate](#decide-candidate)

## Decide Candidate

* accept the candidate only if c8 median useful throughput improves by at least `{{minimum_win}}`
* accept the candidate only if pair-GPU tokens per joule does not regress
* accept the candidate only if p95 completion latency does not regress by more than five percent
* accept the candidate only if no coding-task class loses more than one pass-rate point
* accept the candidate only if perplexity rises by no more than one percent
* accept the candidate only if mean KL divergence rises by no more than `0.01` nat per token
* accept the candidate only if no correctness, cache, fabric, or host-stability gate fails
* if every acceptance condition passes
  * commit the implementation and set `{{baseline}}` to the candidate
* if any acceptance condition fails
  * [Revert Candidate](#revert-candidate)
* [Recurse](#recurse)

## Revert Candidate

* preserve the measurement in the build log
* remove rejected loader paths
* remove rejected runtime flags
* remove rejected additive bytes from both nodes
* restore the last accepted commit
* rebuild the engine
* rerun the focused failure gate
* if the restored gate fails
  * fix the restoration and [Revert Candidate](#revert-candidate)
* [Recurse](#recurse)

## Recurse

* increment `{{iteration}}`
* if `{{iteration}}` reaches `{{max_iterations}}`
  * [Run Final Proof](glm53-concurrent-coding-proof.mdscript.md#run-final-proof)
* if two consecutive iterations improve useful throughput by less than `{{minimum_win}}` and no measured kernel gap remains
  * [Run Final Proof](glm53-concurrent-coding-proof.mdscript.md#run-final-proof)
* [Profile Real Workload](#profile-real-workload)

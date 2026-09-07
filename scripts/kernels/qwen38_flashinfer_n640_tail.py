#!/usr/bin/env python3
"""Materialize Rocket's guarded N640 retained2 FlashInfer source overlay.

The input is FlashInfer commit 91bda04c66f7cb851e1ab3b78b9fecea644b9844
(Apache-2.0). The output supplies the measured c4 path and remains diagnostic
at c8/c16. The pinned dynamic N640 and padded static N768 paths are unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import py_compile
import shutil
from pathlib import Path

FLASHINFER_COMMIT = "91bda04c66f7cb851e1ab3b78b9fecea644b9844"
STATIC_REL = Path("flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_static_kernel.py")
DISPATCH_REL = Path("flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py")
GENERIC_REL = Path(
    "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/generic.py"
)
REFERENCE_SHA256 = {
    STATIC_REL: "c7b6f24b94d7939cc0eb917ab15cef4c34f3dc12bf75e15fc9dc316ee7327f3f",
    DISPATCH_REL: "c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a",
    GENERIC_REL: "3f0be67f2c7f62f20f243baa073af26d9b2a1aff21b1a54eca50961d139b590f",
}


class OverlayContractError(RuntimeError):
    """The pinned source or an exact rewrite anchor drifted."""


def retained_group_count(intermediate: int) -> int:
    if (
        isinstance(intermediate, bool)
        or not isinstance(intermediate, int)
        or intermediate <= 0
        or intermediate % 128
    ):
        raise ValueError("intermediate must be a positive N128 multiple")
    return (intermediate + 255) // 256


def retained_slice_count(intermediate: int, group: int) -> int:
    groups = retained_group_count(intermediate)
    if (
        isinstance(group, bool)
        or not isinstance(group, int)
        or group < 0
        or group >= groups
    ):
        raise ValueError("retained group is outside the exact extent")
    return min(2, intermediate // 128 - group * 2)


def _replace_exact(source: str, old: str, new: str, *, count: int = 1) -> str:
    observed = source.count(old)
    if observed != count:
        raise OverlayContractError(
            f"rewrite anchor count mismatch: expected {count}, observed {observed}"
        )
    return source.replace(old, new)


def _replace_first(source: str, old: str, new: str) -> str:
    if old not in source:
        raise OverlayContractError("rewrite anchor is absent")
    return source.replace(old, new, 1)


def _guard_loop(source: str, start_anchor: str, end_anchor: str) -> str:
    """Guard one retained2 loop without changing either complete pair."""

    start = source.find(start_anchor)
    if start < 0:
        raise OverlayContractError(f"missing loop start anchor: {start_anchor}")
    loop = source.find(
        "for retained_slice_idx in cutlass.range_constexpr(2):", start
    )
    if loop < 0:
        raise OverlayContractError(f"missing retained loop after: {start_anchor}")
    end = source.find(end_anchor, loop)
    if end < 0:
        raise OverlayContractError(f"missing loop end anchor: {end_anchor}")
    line_start = source.rfind("\n", 0, loop) + 1
    body_start = source.find("\n", loop) + 1
    indent = source[line_start:loop]
    if not indent or source[end - 1] != "\n":
        raise OverlayContractError("retained loop indentation drift")
    body = source[body_start:end]
    if not body.strip():
        raise OverlayContractError("retained loop body is empty")
    guarded = (
        source[line_start:body_start]
        + f"{indent}    current_slice = intermediate_slice + Int32(retained_slice_idx)\n"
        + f"{indent}    if current_slice < gate_tile_cnt:\n"
        + "".join("    " + line if line.strip() else line for line in body.splitlines(True))
    )
    return source[:line_start] + guarded + source[end:]


def transform_static(source: str) -> str:
    """Guard the final one-slice group in both compute and DMA pipelines."""

    source = _guard_loop(
        source,
        "# PHASE A: FC1 for this slice (gate + up)",
        "                # ============================================================\n                # PHASE B:",
    )
    source = _guard_loop(
        source,
        "# PHASE B: Sweep ALL FC2 output tiles using cached sA",
        "                    # Scatter using precomputed metadata",
    )
    source = _guard_loop(
        source,
        "# Publish two adjacent N128 FC1 slices",
        "                # FC2 reuses gate B/SFB storage",
    )
    source = _guard_loop(
        source,
        "# Load ALL FC2 tiles continuously",
        "                if Int32(tidx) == Int32(self.tma_load_warp_id * 32):",
    )

    # For a complete pair, retain slice0 in sC and quantize it after slice1's
    # MMA drains. For the N640 tail, slice0 is already the last valid slice, so
    # fence reuse first and quantize the materialized slice into Stage0.
    source = _replace_exact(
        source,
        "if retained_slice_idx == 1:\n                            self.pass_sync_barrier.arrive_and_wait()",
        "if (\n                            retained_slice_idx == 1\n                            or current_slice + Int32(1) == gate_tile_cnt\n                        ):\n                            self.pass_sync_barrier.arrive_and_wait()",
    )
    source = _replace_exact(
        source,
        "if retained_slice_idx == 1:\n                            self.quantize_q1_sC_to_sA_sSFA(",
        "if (\n                            retained_slice_idx == 1\n                            or current_slice + Int32(1) == gate_tile_cnt\n                        ):\n                            self.quantize_q1_sC_to_sA_sSFA(",
    )
    source = _replace_exact(
        source,
        "Int32(1),\n                                epi_rest_m,\n                            )",
        "Int32(retained_slice_idx),\n                                epi_rest_m,\n                            )",
    )
    # The pre-materialization Stage0 quantization is valid only for the second
    # half of a complete pair. The tail's first half has no prior activation.
    source = _replace_exact(
        source,
        "# sC still holds the first slice.  FC1 no longer needs the\n                            # two A/SFA pipeline stages, so quantize it into Stage0.\n                            self.quantize_q1_sC_to_sA_sSFA(",
        "# sC holds slice0 only when this is the second half of a pair.\n                            if retained_slice_idx == 1:\n                                self.quantize_q1_sC_to_sA_sSFA(",
    )
    return source


def transform_dispatch(source: str) -> str:
    source = _replace_exact(
        source,
        "_FORCED_BACKEND: str | None = None",
        "_FORCED_BACKEND: str | None = None\n_EXACT_N640_RETAINED_TAIL = False\n_BARRIER_PHASES = 5\n_BARRIER_TRACE_SLOTS = 256",
    )
    source = _replace_exact(
        source,
        "retained_groups = max(1, n // _STATIC_RETAINED_GROUP_N)",
        "retained_groups = max(1, (n + _STATIC_RETAINED_GROUP_N - 1) // _STATIC_RETAINED_GROUP_N)",
        count=2,
    )
    source = _replace_exact(
        source,
        "    n = _align_up(n, _STATIC_RETAINED_GROUP_N)",
        "    if not (_EXACT_N640_RETAINED_TAIL and n == 640):\n"
        "        n = _align_up(n, _STATIC_RETAINED_GROUP_N)",
    )
    source = _replace_exact(
        source,
        "max_rows * max(1, n // _STATIC_RETAINED_GROUP_N) * k",
        "max_rows\n        * max(\n            1,\n"
        "            (n + _STATIC_RETAINED_GROUP_N - 1)\n"
        "            // _STATIC_RETAINED_GROUP_N,\n"
        "        )\n        * k",
    )
    source = _replace_exact(
        source,
        'if backend == "static" and n % _STATIC_RETAINED_GROUP_N != 0:',
        'if (\n        backend == "static"\n        and n % _STATIC_RETAINED_GROUP_N != 0\n        and not (_EXACT_N640_RETAINED_TAIL and n == 640)\n    ):',
    )
    source = _replace_exact(
        source,
        "    barrier_epoch: torch.Tensor\n\n    # Dynamic-specific",
        "    barrier_epoch: torch.Tensor\n    barrier_phase_clock: torch.Tensor\n\n    # Dynamic-specific",
    )
    source = _replace_exact(
        source,
        "        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),\n        expert_write_rows=",
        "        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),\n        barrier_phase_clock=torch.zeros(\n            (_BARRIER_PHASES, _BARRIER_TRACE_SLOTS, 2),\n            dtype=torch.int64,\n            device=device,\n        ),\n        expert_write_rows=",
    )
    source = _replace_exact(
        source,
        "        barrier_epoch: cute.Tensor,\n        pair_head: cute.Tensor,",
        "        barrier_epoch: cute.Tensor,\n        barrier_phase_clock_ptr: cute.Pointer,\n        pair_head: cute.Tensor,",
    )
    source = _replace_exact(
        source,
        "        task_valid_rows = cute.make_tensor(\n            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))\n        )",
        "        task_valid_rows = cute.make_tensor(\n            task_valid_rows_ptr, layout=cute.make_layout((max_tasks,), stride=(1,))\n        )\n        barrier_phase_clock = cute.make_tensor(\n            barrier_phase_clock_ptr,\n            layout=cute.make_layout(\n                (_BARRIER_PHASES * _BARRIER_TRACE_SLOTS * 2,), stride=(1,)\n            ),\n        )",
    )
    source = _replace_exact(
        source,
        "            barrier_epoch,\n            pair_head,",
        "            barrier_epoch,\n            barrier_phase_clock,\n            pair_head,",
        count=1,
    )
    source = _replace_exact(
        source,
        "    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(\n        cutlass.Int32, (1,), assumed_align=4\n    )",
        "    barrier_epoch_fake = cute.runtime.make_fake_compact_tensor(\n        cutlass.Int32, (1,), assumed_align=4\n    )\n    barrier_phase_clock_fake = make_ptr(\n        cutlass.Int64, 8, cute.AddressSpace.gmem, assumed_align=8\n    )",
    )
    source = _replace_exact(
        source,
        "            barrier_epoch_fake,\n            pair_head_fake,",
        "            barrier_epoch_fake,\n            barrier_phase_clock_fake,\n            pair_head_fake,",
        count=1,
    )
    source = _replace_exact(
        source,
        "        workspace.barrier_epoch,\n        workspace.pair_head,",
        "        workspace.barrier_epoch,\n        workspace.barrier_phase_clock.data_ptr(),\n        workspace.pair_head,",
        count=1,
    )
    return source


def transform_dynamic_instrumentation(source: str) -> str:
    """Add a bounded five-phase clock plane to the generic dynamic kernel."""

    source = _replace_exact(
        source,
        "_PRODUCER_PAIRS_PER_WARP = 2",
        "_PRODUCER_PAIRS_PER_WARP = 2\n_BARRIER_PHASES = 5\n_BARRIER_TRACE_SLOTS = 256",
    )
    clock_helper = '''\n\n@dsl_user_op\ndef _read_globaltimer(*, loc=None, ip=None):\n    return Int64(\n        llvm.inline_asm(\n            T.i64(),\n            [],\n            "mov.u64 $0, %globaltimer;",\n            "=l",\n            has_side_effects=True,\n            is_align_stack=False,\n            asm_dialect=llvm.AsmDialect.AD_ATT,\n        )\n    )\n'''
    source = _replace_exact(source, "\n\nclass DynamicLaunchParams:", clock_helper + "\n\nclass DynamicLaunchParams:")
    source = _replace_exact(
        source,
        "        barrier_epoch: cute.Tensor,\n        grid_x: Int32,",
        "        barrier_epoch: cute.Tensor,\n        barrier_phase_clock: cute.Tensor,\n        phase: Int32,\n        cta: Int32,\n        grid_x: Int32,",
    )
    source = _replace_exact(
        source,
        "            old_epoch = _ld_global_acquire_i32(barrier_epoch_addr)",
        "            trace_base = (phase * Int32(_BARRIER_TRACE_SLOTS) + cta) * Int32(2)\n            barrier_phase_clock[trace_base] = _read_globaltimer()\n            old_epoch = _ld_global_acquire_i32(barrier_epoch_addr)",
    )
    source = _replace_exact(
        source,
        "                _spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)\n        cute.arch.sync_threads()",
        "                _spin_wait_global_eq_i32(barrier_epoch_addr, old_epoch)\n            barrier_phase_clock[trace_base + Int32(1)] = _read_globaltimer()\n        cute.arch.sync_threads()",
    )
    source = _replace_exact(
        source,
        "        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)\n        pair_head:",
        "        barrier_epoch: cute.Tensor,  # [1] int32 (host-zeroed)\n        barrier_phase_clock: cute.Tensor,  # [5,256,2] int64\n        pair_head:",
    )
    source = _replace_exact(
        source,
        "            barrier_epoch,\n            pair_head,",
        "            barrier_epoch,\n            barrier_phase_clock,\n            pair_head,",
        count=1,
    )
    source = _replace_exact(
        source,
        "        barrier_epoch: cute.Tensor,\n        pair_head:",
        "        barrier_epoch: cute.Tensor,\n        barrier_phase_clock: cute.Tensor,\n        pair_head:",
    )
    call = '''self._resident_grid_barrier(\n            barrier_count,\n            barrier_epoch,\n            Int32(gdim_z),\n            is_cta_leader,\n        )'''
    for phase in range(5):
        replacement = f'''self._resident_grid_barrier(\n            barrier_count,\n            barrier_epoch,\n            barrier_phase_clock,\n            Int32({phase}),\n            Int32(bidz),\n            Int32(gdim_z),\n            is_cta_leader,\n        )'''
        source = _replace_first(source, call, replacement)
    return source


def materialize(source_root: Path, output_root: Path) -> None:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if (
        source_root == output_root
        or source_root in output_root.parents
        or output_root in source_root.parents
    ):
        raise OverlayContractError("source and overlay trees must be disjoint")
    for relative, expected in REFERENCE_SHA256.items():
        observed = hashlib.sha256((source_root / relative).read_bytes()).hexdigest()
        if observed != expected:
            raise OverlayContractError(f"pinned FlashInfer source drift: {relative}")
    transforms = {
        STATIC_REL: transform_static,
        DISPATCH_REL: transform_dispatch,
        GENERIC_REL: transform_dynamic_instrumentation,
    }
    transformed = {
        relative: transform((source_root / relative).read_text())
        for relative, transform in transforms.items()
    }
    if output_root.exists():
        if not output_root.is_dir():
            raise OverlayContractError("overlay output is not a directory")
        for relative, expected in transformed.items():
            try:
                observed = (output_root / relative).read_text()
            except OSError as exc:
                raise OverlayContractError("existing overlay is incomplete") from exc
            if observed != expected:
                raise OverlayContractError(f"existing overlay drift: {relative}")
        return
    shutil.copytree(source_root, output_root, copy_function=os.link)
    for relative, transform in transforms.items():
        target = output_root / relative
        target.unlink()
        target.write_text(transformed[relative])
        py_compile.compile(str(target), doraise=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

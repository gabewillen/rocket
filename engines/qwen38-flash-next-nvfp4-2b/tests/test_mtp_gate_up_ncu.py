# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import subprocess
import tempfile
import unittest


ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parents[1]
PROFILE = ENGINE / "bench/mtp_routed_expert_profile.cu"
RUNNER = REPO / "scripts/moe/qwen38-mtp-gate-up-ncu.sh"
SUMMARY = REPO / "scripts/moe/summarize-mtp-gate-up-ncu.py"


class MtpGateUpNcuTest(unittest.TestCase):
    def test_profile_selectors_are_exact_and_default_is_preserved(self) -> None:
        source = PROFILE.read_text()
        self.assertIn('argc > 4 ? argv[4] : "all"', source)
        self.assertIn('selected_case != "c8-k7-r1"', source)
        self.assertIn('selected_case != "c16-k7-r1"', source)
        self.assertIn("!workload.capacity_control", source)
        self.assertIn("workload.capacity_control", source)

    def test_runner_is_fail_closed_and_profiles_only_gate_up(self) -> None:
        runner = RUNNER.read_text()
        for required in (
            "INCOMPLETE\\tvalid=false",
            "--graph-profiling node",
            "regex:.*gate_up_silu.*",
            "--launch-count 1",
            "--clock-control none",
            "--cache-control all",
            "ComputeWorkloadAnalysis",
            "MemoryWorkloadAnalysis",
            "SchedulerStats",
            "WarpStateStats",
            "sudo -n",
            "ROCKET_EXPECTED_HEAD",
            "ROCKET_PEER",
        ):
            self.assertIn(required, runner)

    def test_summary_reports_reuse_and_stalls(self) -> None:
        header = '"Metric Name","Metric Unit","Metric Value"\n'
        rows = (
            '"gpu__time_duration.sum","nsecond","2000000"\n'
            '"smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed","%","25"\n'
            '"gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed","%","50"\n'
            '"lts__throughput.avg.pct_of_peak_sustained_elapsed","%","40"\n'
            '"sm__warps_active.avg.pct_of_peak_sustained_active","%","20"\n'
            '"lts__t_sectors_srcunit_tex_lookup_hit.sum","sector","75"\n'
            '"lts__t_sectors_srcunit_tex_lookup_miss.sum","sector","25"\n'
            '"lts__t_sectors_aperture_device_op_read.sum","sector","100"\n'
            '"lts__t_sectors_aperture_sysmem_op_read.sum","sector","50"\n'
            '"smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct","%","30"\n'
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for case in ("c8-k7-r1", "c16-k7-r1"):
                path = Path(directory) / f"{case}.csv"
                path.write_text(header + rows)
                paths.append(path)
            result = subprocess.run(
                ["python3", str(SUMMARY), *(str(path) for path in paths)],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("tensor issue %", result.stdout)
        self.assertIn("long_scoreboard:30.0%", result.stdout)
        self.assertIn("75.00", result.stdout)
        self.assertIn("c16-k7-r1", result.stdout)


if __name__ == "__main__":
    unittest.main()

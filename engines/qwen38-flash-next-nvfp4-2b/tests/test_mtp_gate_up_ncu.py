# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ENGINE = Path(__file__).resolve().parents[1]
REPO = ENGINE.parents[1]
PROFILE = ENGINE / "bench/mtp_routed_expert_profile.cu"
RUNNER = REPO / "scripts/moe/qwen38-mtp-gate-up-ncu.sh"
SUMMARY = REPO / "scripts/moe/summarize-mtp-gate-up-ncu.py"


class MtpGateUpNcuTest(unittest.TestCase):
    @staticmethod
    def metric_rows() -> list[tuple[str, str, str]]:
        match = re.search(r"^metrics='([^']+)'$", RUNNER.read_text(), re.MULTILINE)
        assert match
        result = []
        for name in match.group(1).split(","):
            if name == "gpu__time_duration.sum":
                result.append((name, "ms", "2.0"))
            elif name.endswith(".pct") or "pct_of_peak" in name:
                value = "30" if "long_scoreboard" in name else "25"
                result.append((name, "%", value))
            else:
                result.append((name, "sector", "100"))
        result.append(("profiler__replayer_passes", "pass", "19"))
        return result

    @staticmethod
    def write_wide(path: Path, rows: list[tuple[str, str, str]], kernel: str = "gate_up_silu") -> None:
        names = ("Kernel Name", *(row[0] for row in rows))
        units = ("", *(row[1] for row in rows))
        values = (kernel, *(row[2] for row in rows))
        path.write_text(
            ",".join(f'"{name}"' for name in names) + "\n" +
            ",".join(f'"{unit}"' for unit in units) + "\n" +
            ",".join(f'"{value}"' for value in values) + "\n"
        )

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
        header = '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        rows = "".join(
            f'"gate_up_silu","{name}","{unit}","{value}"\n'
            for name, unit, value in self.metric_rows()
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
        self.assertIn("50.00", result.stdout)
        self.assertIn("c16-k7-r1", result.stdout)

    def test_summary_accepts_ncu_wide_raw_csv(self) -> None:
        metric_rows = self.metric_rows()
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for case in ("c8-k7-r1", "c16-k7-r1"):
                path = Path(directory) / f"{case}.csv"
                self.write_wide(path, metric_rows)
                paths.append(path)
            result = subprocess.run(
                ["python3", str(SUMMARY), *(str(path) for path in paths)],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertIn("| c8-k7-r1 | 2.000 |", result.stdout)

    def test_summary_rejects_semantic_drift(self) -> None:
        mutations = {
            "duplicate": lambda rows: rows + [rows[0]],
            "missing": lambda rows: rows[:-1],
            "unit": lambda rows: [(rows[0][0], "ns", rows[0][2]), *rows[1:]],
            "locale": lambda rows: [(rows[0][0], rows[0][1], "1,5"), *rows[1:]],
            "nonfinite": lambda rows: [(rows[0][0], rows[0][1], "nan"), *rows[1:]],
            "passes": lambda rows: [
                (name, unit, "20" if name == "profiler__replayer_passes" else value)
                for name, unit, value in rows
            ],
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                paths = [Path(directory) / f"{case}.csv" for case in ("c8-k7-r1", "c16-k7-r1")]
                self.write_wide(paths[0], mutate(self.metric_rows()))
                self.write_wide(paths[1], self.metric_rows())
                result = subprocess.run(
                    ["python3", str(SUMMARY), *(str(path) for path in paths)],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)

        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{case}.csv" for case in ("c8-k7-r1", "c16-k7-r1")]
            self.write_wide(paths[0], self.metric_rows(), kernel="wrong_kernel")
            self.write_wide(paths[1], self.metric_rows())
            result = subprocess.run(
                ["python3", str(SUMMARY), *(str(path) for path in paths)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_summary_rejects_mixed_schemas_and_wrong_cases(self) -> None:
        rows = self.metric_rows()
        with tempfile.TemporaryDirectory() as directory:
            c8 = Path(directory) / "c8-k7-r1.csv"
            c16 = Path(directory) / "c16-k7-r1.csv"
            self.write_wide(c8, rows)
            c16.write_text(
                '"Kernel Name","Metric Name","Metric Unit","Metric Value"\n' +
                "".join(
                    f'"gate_up_silu","{name}","{unit}","{value}"\n'
                    for name, unit, value in rows
                )
            )
            mixed = subprocess.run(
                ["python3", str(SUMMARY), str(c8), str(c16)],
                capture_output=True,
                text=True,
            )
            wrong = subprocess.run(
                ["python3", str(SUMMARY), str(c8), str(c8)],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(mixed.returncode, 0)
        self.assertNotEqual(wrong.returncode, 0)


if __name__ == "__main__":
    unittest.main()

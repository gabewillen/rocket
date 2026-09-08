import copy
import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/moe/qwen38-router-cohort-launch.py"
MANIFEST = ROOT / "scripts/moe/qwen38-router-cohort-launch-v4.json"
SPEC = importlib.util.spec_from_file_location("router_launch", LAUNCHER)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LaunchContractTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.plan = MODULE.build_plan(
            self.manifest, Path("/home/tester"), Path("/home/tester/calibration"),
            50181, "d9643b0-r4",
        )

    def test_matches_captured_successful_entrypoint_and_bind_contract(self):
        for node in ("head", "worker"):
            plan = self.plan["nodes"][node]
            argv = plan["docker_argv"]
            entrypoint = argv.index("--entrypoint")
            self.assertEqual(argv[entrypoint + 1], "python3")
            self.assertEqual(
                [item["destination"] for item in plan["binds"]],
                self.manifest["nodes"][node]["bind_destination_order"],
            )
            parent = self.manifest["nodes"][node]["bind_destination_order"].index(
                "/root/.cache/huggingface"
            )
            destinations = self.manifest["nodes"][node]["bind_destination_order"]
            self.assertLess(parent, destinations.index(f"{MODULE.SNAPSHOT}/config.json"))
            self.assertLess(
                parent, destinations.index(f"{MODULE.SNAPSHOT}/hf_quant_config.json")
            )

    def test_worker_uses_authenticated_named_cache_volume(self):
        cache = self.plan["nodes"]["worker"]["binds"][5]
        self.assertEqual(cache["source"], "vllm-fn-hf")
        self.assertEqual(cache["destination"], "/root/.cache/huggingface")

    def test_rejects_missing_explicit_entrypoint(self):
        broken = copy.deepcopy(self.plan["nodes"]["head"])
        index = broken["docker_argv"].index("--entrypoint")
        del broken["docker_argv"][index : index + 2]
        with self.assertRaises(ValueError):
            MODULE.validate_node_plan("head", broken, self.manifest)

    def test_rejects_reordered_parent_and_child_mounts(self):
        broken = copy.deepcopy(self.plan["nodes"]["worker"])
        bind_indexes = [
            index for index, item in enumerate(broken["docker_argv"]) if item == "-v"
        ]
        first = bind_indexes[5] + 1
        second = bind_indexes[6] + 1
        broken["docker_argv"][first], broken["docker_argv"][second] = (
            broken["docker_argv"][second], broken["docker_argv"][first]
        )
        with self.assertRaisesRegex(ValueError, "argv bind ordering"):
            MODULE.validate_node_plan("worker", broken, self.manifest)

    def test_torchrun_workload_is_fixed(self):
        for node, rank in (("head", 0), ("worker", 1)):
            argv = self.plan["nodes"][node]["docker_argv"]
            self.assertIn(f"--node-rank={rank}", argv)
            self.assertIn("--master-port=50181", argv)
            self.assertEqual(argv[-9:], [
                "/work/qwen38-router-cohort-live.py", "--concurrency", "16",
                "--decode", "24", "--prefix-tokens", "8192",
                "--divergence-tokens", "128",
            ])


if __name__ == "__main__":
    unittest.main()

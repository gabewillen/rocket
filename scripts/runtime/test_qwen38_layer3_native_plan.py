# SPDX-License-Identifier: Apache-2.0
import argparse
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def seal(value):
    value.pop("descriptor_sha256", None)
    value["descriptor_sha256"] = hashlib.sha256(
        canonical(value).encode()
    ).hexdigest()
    return canonical(value) + "\n"


def run(binary: Path, plan: Path, rank: int, accepted: bool):
    result = subprocess.run([str(binary), str(plan), str(rank)],
                            capture_output=True, text=True, check=False)
    if (result.returncode == 0) != accepted:
        raise RuntimeError(result.stdout + result.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.plan.read_bytes())
    rank = source["rank"]
    run(args.binary, args.plan, rank, True)
    mutations = {}
    value = copy.deepcopy(source); value.pop("layout_sha256")
    mutations["missing_top_level"] = value
    value = copy.deepcopy(source); value["unknown_top_level"] = 1
    mutations["unknown_top_level"] = value
    value = copy.deepcopy(source); value["extents"][0]["offset_bytes"] = value["slab_bytes"]
    mutations["out_of_bounds"] = value
    value = copy.deepcopy(source)
    target = [item for item in value["extents"] if item["storage"] == "target_slab"]
    target[1]["offset_bytes"] = target[0]["offset_bytes"]
    target[1]["source_chunks"] = copy.deepcopy(target[0]["source_chunks"])
    mutations["overlap"] = value
    value = copy.deepcopy(source); value["extents"][0]["strides"][-1] = 2
    mutations["stride"] = value
    value = copy.deepcopy(source); value["extents"][0].pop("abi")
    value["extents"][0]["unknown"] = "native"
    mutations["unknown_extent_field"] = value
    value = copy.deepcopy(source); value["pair_reduce"].pop("calls")
    value["pair_reduce"]["unknown"] = 70
    mutations["unknown_pairreduce_field"] = value
    value = copy.deepcopy(source); value["rank"] = 1 << 40
    mutations["rank_narrowing"] = value
    value = copy.deepcopy(source); value["peer_rank"] = 1 << 40
    mutations["peer_rank_narrowing"] = value
    value = copy.deepcopy(source)
    value["schema"] = value["schema"] + "\0evil"
    mutations["escaped_nul_string"] = value
    value = copy.deepcopy(source)
    value["pair_reduce"]["rails"][0] += "\0evil"
    mutations["escaped_nul_array_string"] = value
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, mutation in mutations.items():
            path = root / f"{name}.json"
            path.write_text(seal(mutation))
            run(args.binary, path, rank, False)
        digest = root / "digest_mismatch.json"
        raw = args.plan.read_text()
        digest.write_text(raw.replace('"compare_row":34', '"compare_row":33', 1))
        run(args.binary, digest, rank, False)
        mutations["digest_mismatch"] = None
        embedded_nul = root / "embedded_nul.json"
        embedded_nul.write_bytes(
            args.plan.read_bytes()[:-1]
            + b'\0{"unknown_after_nul":true}\n'
        )
        run(args.binary, embedded_nul, rank, False)
        mutations["embedded_nul"] = None
        trailing_json = root / "trailing_json.json"
        trailing_json.write_bytes(args.plan.read_bytes()[:-1] + b'{}\n')
        run(args.binary, trailing_json, rank, False)
        mutations["trailing_json"] = None
    print(json.dumps({"schema": "rocket.qwen38.layer3-native-plan-test.v1",
                      "valid": True, "mutations_rejected": sorted(mutations)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

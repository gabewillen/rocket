#!/usr/bin/env python3
"""Reduce legacy or v2 Qwen3.8 activation telemetry into checked JSON."""

import argparse
import json
import re
import sys

LEGACY_LINE = re.compile(
    r"ROCKET_NVFP4_CALIBRATION\t"
    r"(layer\.(\d+)\.linear_attn\.(in_proj_qkvz|in_proj_ba|out_proj))\t"
    r"([0-9.eE+-]+)"
)
V2_PREFIX = "ROCKET_NVFP4_TELEMETRY\t"
V2_REQUIRED_STATS = {
    "source_numel",
    "sample_numel",
    "absmax",
    "mean",
    "rms",
    "abs_p50",
    "abs_p90",
    "abs_p99",
    "histogram_log2",
}


def parse_stream(lines):
    maxima = {}
    telemetry = {}
    for line in lines:
        match = LEGACY_LINE.search(line)
        if match:
            name, _, _, raw_value = match.groups()
            value = float(raw_value)
            maxima[name] = max(value, maxima.get(name, 0.0))
        marker = line.find(V2_PREFIX)
        if marker < 0:
            continue
        try:
            record = json.loads(line[marker + len(V2_PREFIX) :])
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid v2 telemetry JSON: {error}") from error
        if record.get("schema") != "rocket.qwen38.activation-telemetry.v2":
            raise ValueError(f"unsupported telemetry schema: {record.get('schema')!r}")
        channel = record.get("channel")
        call = record.get("call")
        if not isinstance(channel, str) or not isinstance(call, int):
            raise ValueError("v2 telemetry requires string channel and integer call")
        missing = V2_REQUIRED_STATS - record.keys()
        if missing:
            raise ValueError(f"v2 telemetry missing fields: {sorted(missing)}")
        if (
            not isinstance(record["histogram_log2"], list)
            or len(record["histogram_log2"]) != 10
        ):
            raise ValueError("v2 telemetry histogram_log2 must contain 10 bins")
        if not 0 < record["sample_numel"] <= record["source_numel"]:
            raise ValueError("v2 telemetry sample_numel is outside source bounds")
        previous = telemetry.get(channel)
        if previous is None or call >= previous["call"]:
            telemetry[channel] = record
    return maxima, telemetry


def legacy_layers(maxima):
    by_layer = {}
    for name, value in sorted(maxima.items()):
        _, layer, _, projection = name.split(".")
        by_layer.setdefault(layer, {})[projection] = value
    complete = [layer for layer, values in by_layer.items() if len(values) == 3]
    return by_layer, complete


def layer_count(telemetry, suffix):
    return len(
        {
            channel.split(".")[1]
            for channel in telemetry
            if channel.endswith(suffix) and channel.startswith("layer.")
        }
    )


def channel_count(telemetry, fragment):
    return sum(fragment in channel for channel in telemetry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=36)
    parser.add_argument("--require-expanded", action="store_true")
    parser.add_argument(
        "--expanded-v2-only",
        action="store_true",
        help="gate expanded coverage from v2 records without requiring legacy maxima",
    )
    parser.add_argument(
        "--min-emission-call",
        type=int,
        help="ignore v2 records below this emission call (required with --expanded-v2-only)",
    )
    parser.add_argument("--full-attention-layers", type=int, default=12)
    parser.add_argument("--ple-layers", type=int)
    parser.add_argument("--router-layers", type=int)
    parser.add_argument("--recurrent-state-layers", type=int)
    args = parser.parse_args()
    if args.expanded_v2_only:
        if not args.require_expanded:
            parser.error("--expanded-v2-only requires --require-expanded")
        if args.min_emission_call is None or args.min_emission_call < 1:
            parser.error(
                "--expanded-v2-only requires --min-emission-call >= 1"
            )
    elif args.min_emission_call is not None:
        parser.error("--min-emission-call requires --expanded-v2-only")
    try:
        maxima, telemetry = parse_stream(sys.stdin)
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

    _, legacy_complete = legacy_layers(maxima)
    expected_channels = args.layers * 3
    if not args.expanded_v2_only and (
        len(legacy_complete) != args.layers or len(maxima) != expected_channels
    ):
        print(
            f"incomplete calibration: {len(legacy_complete)}/{args.layers} layers, "
            f"{len(maxima)}/{expected_channels} channels",
            file=sys.stderr,
        )
        raise SystemExit(1)

    minimum_call = args.min_emission_call or 1
    gated_telemetry = {
        channel: record
        for channel, record in telemetry.items()
        if record["call"] >= minimum_call
    }
    v2_linear_layers = layer_count(gated_telemetry, ".linear_attn.output")

    coverage = {
        "linear_attention_layers": (
            v2_linear_layers if args.expanded_v2_only else len(legacy_complete)
        ),
        "linear_projection_input_channels": sum(
            channel.endswith(".input")
            and any(
                marker in channel
                for marker in (
                    ".linear_attn.in_proj_qkvz.",
                    ".linear_attn.in_proj_ba.",
                    ".linear_attn.out_proj.",
                )
            )
            for channel in gated_telemetry
        ),
        "linear_projection_output_channels": sum(
            channel.endswith(".output")
            and any(
                marker in channel
                for marker in (
                    ".linear_attn.in_proj_qkvz.",
                    ".linear_attn.in_proj_ba.",
                    ".linear_attn.out_proj.",
                )
            )
            for channel in gated_telemetry
        ),
        "full_attention_layers": layer_count(
            gated_telemetry, ".full_attn.output"
        ),
        "full_qkv_projection_layers": layer_count(
            gated_telemetry, ".full_attn.qkv_proj.output"
        ),
        "full_output_projection_layers": layer_count(
            gated_telemetry, ".full_attn.o_proj.output"
        ),
        "ple_layers": layer_count(gated_telemetry, ".ple.output"),
        "router_layers": layer_count(
            gated_telemetry, ".router.topk.output"
        ),
        "recurrent_state_layers": layer_count(
            gated_telemetry, ".linear_attn.recurrent_state.output"
        ),
    }
    if args.expanded_v2_only:
        coverage["legacy_linear_attention_layers"] = len(legacy_complete)
        coverage["legacy_channels"] = len(maxima)
    if args.require_expanded:
        requirements = {
            "linear_projection_input_channels": args.layers * 3,
            "linear_projection_output_channels": args.layers * 3,
            "full_attention_layers": args.full_attention_layers,
            "full_qkv_projection_layers": args.full_attention_layers,
            "full_output_projection_layers": args.full_attention_layers,
            "ple_layers": args.ple_layers,
            "router_layers": args.router_layers,
            "recurrent_state_layers": args.recurrent_state_layers,
        }
        failures = [
            f"{name}={coverage[name]}/{expected}"
            for name, expected in requirements.items()
            if expected is not None and coverage[name] != expected
        ]
        if not gated_telemetry:
            failures.append("v2 telemetry absent")
        if failures:
            print("incomplete expanded telemetry: " + ", ".join(failures), file=sys.stderr)
            raise SystemExit(1)

    result = {
        "schema": "rocket.qwen38.activation-summary.v2",
        "legacy_schema": "rocket.qwen38.activation-maxima.v1",
        "coverage": coverage,
        "channels": maxima,
        "telemetry": gated_telemetry,
    }
    if args.expanded_v2_only:
        result["gate"] = {
            "source": "v2_only",
            "min_emission_call": minimum_call,
        }
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()

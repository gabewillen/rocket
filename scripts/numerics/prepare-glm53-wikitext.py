#!/usr/bin/env python3
"""Create a deterministic line-oriented WikiText validation scoring corpus."""

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

URL = "https://datasets-server.huggingface.co/rows"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--minimum-characters", type=int, default=100)
    parser.add_argument("--scan", type=int, default=100)
    args = parser.parse_args()
    query = urllib.parse.urlencode({
        "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "split": "validation",
        "offset": 0,
        "length": args.scan,
    })
    with urllib.request.urlopen(f"{URL}?{query}") as response:
        payload = json.load(response)
    rows = []
    for item in payload["rows"]:
        text = " ".join(item["row"]["text"].split())
        if len(text) >= args.minimum_characters:
            rows.append(text)
        if len(rows) == args.rows:
            break
    if len(rows) != args.rows:
        raise RuntimeError(f"found {len(rows)} qualifying rows, wanted {args.rows}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(rows) + "\n")
    print(json.dumps({"out": str(args.out), "rows": len(rows),
                      "characters": sum(map(len, rows)), "source": f"{URL}?{query}"}))


if __name__ == "__main__":
    main()

"""Stage 2: export dataset questions for an external retrieval project."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import load_qa_dataset, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("trivia", "nq", "pop", "strategy"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts, _ = load_qa_dataset(args.dataset, args.dataset_root or args.root / "datasets")
    output = args.output or args.root / "retrieval_queries" / f"{args.dataset}.jsonl"
    write_jsonl(output, [{"idx": idx, "query": prompt} for idx, prompt in enumerate(prompts)])
    print(f"Saved {len(prompts)} retrieval queries to {output}")


if __name__ == "__main__":
    main()


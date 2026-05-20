"""Stage 1: answer questions without retrieval and split them by correctness."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import load_qa_dataset, question_from_prompt, write_json
from llm import generate_texts, judge_answers, make_llm, model_id, unload_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("trivia", "nq", "pop", "strategy"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts, answers = load_qa_dataset(args.dataset, args.dataset_root or args.root / "datasets")

    llm = make_llm(
        model_id(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )
    try:
        generations = generate_texts(llm, args.model, prompts)
    finally:
        unload_llm(llm)

    rows = [
        {
            "idx": idx,
            "query": question_from_prompt(prompt),
            "ground_truth": answer if isinstance(answer, list) else [answer],
            "generated_text": generation,
        }
        for idx, (prompt, answer, generation) in enumerate(zip(prompts, answers, generations, strict=True))
    ]
    judged = judge_answers(rows, tensor_parallel_size=args.tensor_parallel_size)

    correct = [row for row in judged if row["prediction"] == "correct"]
    incorrect = [row for row in judged if row["prediction"] != "correct"]
    split_dir = args.root / "datasets" / "splits" / args.dataset
    write_json(split_dir / f"{args.model}_correct.json", correct)
    write_json(split_dir / f"{args.model}_incorrect.json", incorrect)
    print(f"Saved {len(correct)} correct and {len(incorrect)} incorrect rows to {split_dir}")


if __name__ == "__main__":
    main()


"""Stage 4: answer with retrieval settings and judge correctness."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from understanding_rag.config import MAX_RETRIEVED_DOCS, RANDOM_SEED, RETRIEVAL_SETTINGS
from understanding_rag.data import load_qa_dataset, question_from_prompt, read_json, read_jsonl, write_json
from understanding_rag.llm import generate_texts, judge_answers, make_llm, model_id, unload_llm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("trivia", "nq", "pop", "strategy"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--retrieved-jsonl", type=Path, default=None, help="Required for faithful all_20 prompts.")
    parser.add_argument("--settings", nargs="+", default=list(RETRIEVAL_SETTINGS))
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=18000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def text(doc: dict[str, Any]) -> str:
    return doc.get("retrieval text") or doc.get("text") or doc.get("contents") or ""


def selected_docs(setting: str, row: dict[str, Any], all_docs: list[dict[str, Any]] | None) -> list[list[str]]:
    if setting in ("relevant", "distracting", "random"):
        return [[text(doc)] for doc in row[setting]]
    if setting == "relevant_3_distracting":
        distractors = [text(doc) for doc in row["distracting"][:3]]
        if len(distractors) < 3:
            return []
        return [[text(doc), *distractors] for doc in row["relevant"]]
    if setting == "relevant_3_random":
        random_docs = [text(doc) for doc in row["random"][:3]]
        if len(random_docs) < 3:
            return []
        return [[text(doc), *random_docs] for doc in row["relevant"]]
    if setting == "all_20":
        docs = all_docs[:MAX_RETRIEVED_DOCS] if all_docs is not None else row["relevant"] + row["distracting"] + row["random"]
        return [[text(doc) for doc in docs[:MAX_RETRIEVED_DOCS]]]
    raise ValueError(f"Unknown setting {setting!r}")


def build_rows(
    setting: str,
    ids: set[int],
    prompts: list[str],
    answers: list[list[str]],
    retrieval_doc: list[dict[str, Any]],
    retrieved_by_idx: dict[int, list[dict[str, Any]]],
) -> tuple[list[str], list[dict[str, Any]]]:
    prompt_texts, rows = [], []
    for row in retrieval_doc:
        idx = row["idx"]
        if idx not in ids:
            continue
        docs_for_setting = selected_docs(setting, row, retrieved_by_idx.get(idx))
        for docs in docs_for_setting:
            random.shuffle(docs)
            question = question_from_prompt(prompts[idx])
            retrieval_text = "\n\n".join(docs)
            prompt = (
                "Based on the following documents, answer the following question briefly and concisely.\n\n"
                f"Documents:\n\n{retrieval_text}\n\nQuestion: {question}"
            )
            prompt_texts.append(prompt)
            rows.append(
                {
                    "idx": idx,
                    "query": question,
                    "doc": docs if len(docs) > 1 else docs[0],
                    "ground_truth": answers[idx] if isinstance(answers[idx], list) else [answers[idx]],
                }
            )
    return prompt_texts, rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    prompts, answers = load_qa_dataset(args.dataset, args.dataset_root or args.root / "datasets")
    retrieval_doc = read_json(args.root / "retrieval_doc" / f"{args.dataset}.json")
    retrieved_by_idx = {}
    if args.retrieved_jsonl:
        retrieved_by_idx = {idx: row["ctxs"] for idx, row in enumerate(read_jsonl(args.retrieved_jsonl))}

    split_dir = args.root / "datasets" / "splits" / args.dataset
    splits = {
        "correct": {row["idx"] for row in read_json(split_dir / f"{args.model}_correct.json")},
        "incorrect": {row["idx"] for row in read_json(split_dir / f"{args.model}_incorrect.json")},
    }

    llm = make_llm(model_id(args.model), args.tensor_parallel_size, args.max_model_len)
    generated_by_split_setting = {}
    try:
        for difficulty, ids in splits.items():
            for setting in args.settings:
                prompt_texts, rows = build_rows(setting, ids, prompts, answers, retrieval_doc, retrieved_by_idx)
                generations = generate_texts(llm, args.model, prompt_texts)
                generated_by_split_setting[(difficulty, setting)] = [
                    {**row, "generated_text": generation}
                    for row, generation in zip(rows, generations, strict=True)
                ]
    finally:
        unload_llm(llm)

    for (difficulty, setting), rows in generated_by_split_setting.items():
        judged = judge_answers(rows, tensor_parallel_size=args.tensor_parallel_size)
        output = args.root / "rag_outputs" / args.dataset / args.model / difficulty / f"{setting}.json"
        write_json(output, judged)
        print(f"Saved {len(judged)} rows to {output}")


if __name__ == "__main__":
    main()


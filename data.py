"""Dataset loading and common JSON helpers."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any

from datasets import load_dataset

QUESTION_PROMPT = "Answer the following question briefly and concisely.\n\nQuestion: {question}\n\nAnswer:"

JUDGE_PROMPT = """You are an impartial evaluator tasked with judging the correctness of an answer.

### Question
{question}

### Ground Truth Answers
{ground_truths}

### Model Output
{model_output}

Your task: Determine if the model output is semantically and logically consistent with any of the ground truth answers.

If it conveys the same meaning or correct information (even with different wording), mark it as correct.

For yes/no questions, you only need to check whether the final [yes/no] prediction aligns with the ground truth or not.

If the question is about the date, you need to check the consistency of year, month, and day. It is OK if the model outputs the year only, but it is unacceptable if the generated one does not match the ground truth.

If the model abstain to answer the question (e.g., saying the document does not contain sufficient information to answer the question), mark it as abstain.

Respond ONLY with one of the following JSON objects:

{{"verdict": "correct"}}
{{"verdict": "incorrect"}}
{{"verdict": "abstain"}}
"""


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def question_from_prompt(prompt: str) -> str:
    if "\n\nQuestion: " not in prompt:
        return prompt
    return prompt.split("\n\nQuestion: ", 1)[1].removesuffix("\n\nAnswer:")


def _trivia(dataset_root: Path) -> tuple[list[str], list[list[str]]]:
    path = dataset_root / "triviaqa" / "qa" / "wikipedia-dev.json"
    data = read_json(path)
    prompts, answers = [], []
    for item in data["Data"]:
        aliases = item["Answer"]["Aliases"]
        normalized = item["Answer"].get("NormalizedValue", "")
        all_answers = [normalized] + aliases if normalized else aliases
        prompts.append(QUESTION_PROMPT.format(question=item["Question"]))
        answers.append(list(dict.fromkeys(answer for answer in all_answers if answer)))
    return prompts, answers


def _nq(dataset_root: Path) -> tuple[list[str], list[list[str]]]:
    candidates = sorted((dataset_root / "NQ" / "dev").glob("nq-dev-*.jsonl.gz"))
    if not candidates:
        raise FileNotFoundError(f"No NQ dev shards found under {dataset_root / 'NQ' / 'dev'}")

    prompts, answers = [], []
    with gzip.open(candidates[0], "rt", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            short_answers = []
            annotations = item.get("annotations") or []
            if annotations:
                annotation = annotations[0]
                for answer in annotation.get("short_answers", []):
                    tokens = item["document_tokens"][answer["start_token"] : answer["end_token"]]
                    text = " ".join(token["token"] for token in tokens if not token.get("html_token"))
                    if text:
                        short_answers.append(text)
                if not short_answers and annotation.get("yes_no_answer") not in (None, "NONE"):
                    short_answers.append(annotation["yes_no_answer"])
            if short_answers:
                prompts.append(QUESTION_PROMPT.format(question=item["question_text"]))
                answers.append(short_answers)
    return prompts, answers


def _pop(_: Path) -> tuple[list[str], list[list[str]]]:
    prompts, answers = [], []
    for item in load_dataset("akariasai/PopQA", split="test"):
        possible_answers = item["possible_answers"]
        if isinstance(possible_answers, str):
            possible_answers = json.loads(possible_answers)
        prompts.append(QUESTION_PROMPT.format(question=item["question"]))
        answers.append(possible_answers)
    return prompts, answers


def _strategy(_: Path) -> tuple[list[str], list[list[str]]]:
    prompts, answers = [], []
    for item in load_dataset("ChilleD/StrategyQA", split="test"):
        answer = "yes" if item["answer"] else "no"
        prompts.append(QUESTION_PROMPT.format(question=item["question"]))
        answers.append([answer, answer.capitalize(), "true" if item["answer"] else "false"])
    return prompts, answers


DATASET_LOADERS = {
    "trivia": _trivia,
    "nq": _nq,
    "pop": _pop,
    "strategy": _strategy,
}


def load_qa_dataset(dataset: str, dataset_root: str | Path | None = None) -> tuple[list[str], list[list[str]]]:
    if dataset not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset {dataset!r}. Expected one of: {sorted(DATASET_LOADERS)}")
    root = Path(dataset_root or os.environ.get("RAG_DATASET_ROOT", "datasets"))
    return DATASET_LOADERS[dataset](root)


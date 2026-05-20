"""Stage 5: extract last prompt-token representations for each retrieval setting."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BASE_MODEL_TO_INSTRUCTION_TOKENIZER, MODELS, RETRIEVAL_SETTINGS
from data import read_json, read_jsonl, write_json
from llm import is_base_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("trivia", "nq", "pop", "strategy"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--difficulty", required=True, choices=("correct", "incorrect"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--settings", nargs="+", default=["no_doc", *RETRIEVAL_SETTINGS])
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def build_input(tokenizer: AutoTokenizer, model_name: str, question: str, docs: str | list[str] | None = None) -> torch.Tensor:
    if docs is not None:
        doc_text = "\n\n".join(docs) if isinstance(docs, list) else docs
        prompt = (
            "Based on the following documents, answer the following question briefly and concisely.\n\n"
            f"Documents:\n\n{doc_text}\n\nQuestion: {question}"
        )
    else:
        prompt = f"Answer the following question briefly and concisely. Question: {question}"

    if is_base_model(model_name):
        return tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")

    input_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer([input_text], return_tensors="pt").input_ids.to("cuda")


def normalize_hidden_states(model: AutoModelForCausalLM, model_name: str, hidden_states: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if "gemma3-27B" in model_name:
        return torch.stack([model.model.language_model.norm(layer[0]) for layer in hidden_states], dim=0)
    return torch.stack([model.model.norm(layer[0]) for layer in hidden_states], dim=0)


def record_representation(model: AutoModelForCausalLM, model_name: str, input_ids: torch.Tensor, row: dict[str, Any]) -> dict[str, Any]:
    with torch.no_grad():
        outputs = model(input_ids=input_ids, return_dict=True, output_hidden_states=True)
        hidden_states = outputs["hidden_states"][1:]
        norm_res = normalize_hidden_states(model, model_name, hidden_states)
        origin_res = torch.stack([layer[0] for layer in hidden_states], dim=0)

    with autocast(dtype=torch.float):
        return {
            "idx": row["idx"],
            "prediction": row.get("prediction", "unknown"),
            "rep": norm_res[:, -1, :].cpu(),
            "origin_rep": origin_res[:, -1, :].cpu(),
            "rep_mean": norm_res.mean(dim=1).cpu(),
            "origin_rep_mean": origin_res.mean(dim=1).cpu(),
        }


def main() -> None:
    args = parse_args()
    if args.model not in MODELS:
        raise ValueError(f"Unknown model {args.model!r}. Expected one of: {sorted(MODELS)}")

    tokenizer_model = MODELS[BASE_MODEL_TO_INSTRUCTION_TOKENIZER.get(args.model, args.model)]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[args.model],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval().requires_grad_(False)

    output_dir = args.output_dir or args.root / "prompt_rep" / args.dataset / args.model / args.difficulty
    output_dir.mkdir(parents=True, exist_ok=True)
    query_rows = read_jsonl(args.root / "retrieval_queries" / f"{args.dataset}.jsonl")
    split_rows = read_json(args.root / "datasets" / "splits" / args.dataset / f"{args.model}_{args.difficulty}.json")

    for setting in args.settings:
        if setting == "no_doc":
            rows = split_rows
        else:
            rows = read_json(args.root / "rag_outputs" / args.dataset / args.model / args.difficulty / f"{setting}.json")

        output = []
        for row in tqdm(rows, desc=setting):
            if setting == "no_doc":
                query = query_rows[row["idx"]]["query"].split("\n\nQuestion: ", 1)[-1].removesuffix("\n\nAnswer:")
                input_ids = build_input(tokenizer, args.model, query)
            else:
                input_ids = build_input(tokenizer, args.model, row["query"], row["doc"])
            output.append(record_representation(model, args.model, input_ids, row))

        torch.save(output, output_dir / f"{setting}.pt")

    write_json(output_dir / "manifest.json", {"dataset": args.dataset, "model": args.model, "difficulty": args.difficulty})
    print(f"Saved prompt representations to {output_dir}")


if __name__ == "__main__":
    main()


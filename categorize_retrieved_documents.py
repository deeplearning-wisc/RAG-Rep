"""Stage 3: label retrieved documents and build relevant/distracting/random doc sets."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from openai import AzureOpenAI
from tqdm import tqdm

from config import MAX_RETRIEVED_DOCS, RANDOM_SEED
from data import load_qa_dataset, read_jsonl, write_json

SYSTEM_PROMPT = """You are an objective evidence classifier. Given a user question (query), a list of possible answers (multiple choice or yes/no), and a single document, decide whether the document is **relevant**, **distracting**, or **neutral** with respect to answering the query.

- Do NOT produce chain-of-thought. Provide only the required structured output (JSON, see schema below) and a concise 1-2 sentence rationale (no internal reasoning steps).
- Use external/world knowledge only to determine whether a document implicitly supports an answer via ordinary inference (e.g., a fact that implies solvability, gender, date, etc.). Do not invent or hallucinate facts that are not in the document when justifying the label.

- Follow the definitions and heuristics below exactly.

## Required OUTPUT (strict JSON)

Return a single JSON object with these fields only:

```
{
  "label": "relevant" | "distracting" | "neutral",
  "confidence": 0.00-1.00,
  "rationale": "<one- or two-sentence justification (no chain-of-thought)>",
  "supporting_spans": ["<short excerpt(s) from the document that justify the judgment>"],
  "inference_type": "direct" | "indirect" | "multi-hop" | "contradiction" | "mixed"
}
```

- label: one of relevant, distracting, neutral (must match exactly).
- confidence: numeric 0-1 reflecting how certain the label is (see scoring guidance below).
- rationale: no more than two sentences, explaining why label was chosen (no internal chain-of-thought).
- supporting_spans: zero or more short text snippets (≤ 2 lines each) taken verbatim from the document that most strongly support the label. If none, return [].
- inference_type:
    - direct — the document explicitly states the answer (or exact candidate).
    - indirect — the document gives facts that strongly imply the answer (single-step inference).
    - multi-hop — the document provides an intermediate hop (necessary fact) that, combined with other known facts, supports the answer.
    - contradiction — the document asserts facts that contradict the correct answer.
    - mixed — document contains both supporting and contradictory statements.

## Label definitions & heuristics (use these to make the decision)

### RELEVANT
- The correct answer (or parts of answer) is directly appeared in the document → direct.
- OR the document contains facts that clearly support the correct answer either by single-step inference (indirect) or by providing a necessary intermediate hop for a multi-hop inference (multi-hop). If Gold_answer is provided, evaluate support relative to it.
- If the doc contains intermediate facts that are required to get to the final answer (even though the final answer is not present), treat it as relevant (set inference_type = "multi-hop").
- Provide supporting_spans identifying the explicit sentence(s) or fact(s).

### DISTRACTING
- The document asserts claims that would lead a reader away from the correct answer (i.e., it contradicts the correct answer or makes claims that support an incorrect candidate). Use inference_type = "contradiction" if it explicitly contradicts.
- Or the document contains plausible but misleading facts that do not support the correct answer and could plausibly be mistaken for support (e.g., plausible but irrelevant facts presented as if they answer the question). Return supporting spans that illustrate the misleading claim.
- Or the document discuss other things that are related to some entities in the query, while does not provide hints for a reader to answer the question.

### NEUTRAL
- The document is unrelated to the query.

## Confidence scoring guidance (suggested mapping)
- >= 0.90: explicit textual statement of the answer or a clear contradiction/distriction.
- 0.75 - 0.89: strong indirect support (single-step inference) or a strong but not explicit contradiction/distriction.
- 0.55 - 0.74: moderate evidence (document gives facts that imply answer but not overwhelmingly).
- 0.30 - 0.54: weak or partial evidence, or small inconsistency; label should be conservative.
- <= 0.29: little or no evidence; use for neutral decisions.

Set a numeric value according to this guidance.
"""

USER_PROMPT = """Query: {query}
Correct Answer(s): {answers}
Document: {document}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("trivia", "nq", "pop", "strategy"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--retrieved-jsonl", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=MAX_RETRIEVED_DOCS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--azure-endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--azure-api-key", default=os.environ.get("AZURE_OPENAI_API_KEY"))
    parser.add_argument("--azure-api-version", default=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"))
    parser.add_argument("--classifier-model", default=os.environ.get("AZURE_OPENAI_CLASSIFIER_MODEL", "gpt-5"))
    return parser.parse_args()


def doc_text(doc: dict[str, Any]) -> str:
    return doc.get("retrieval text") or doc.get("text") or doc.get("contents") or ""


def classify_document(client: AzureOpenAI, model: str, query: str, answers: list[str], document: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(query=query, answers=json.dumps(answers), document=document)},
        ],
        max_completion_tokens=1000,
        model=model,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def main() -> None:
    args = parse_args()
    if not args.azure_endpoint or not args.azure_api_key:
        raise ValueError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or pass --azure-endpoint/--azure-api-key.")

    random.seed(args.seed)
    prompts, answers = load_qa_dataset(args.dataset, args.dataset_root or args.root / "datasets")
    retrieved = read_jsonl(args.retrieved_jsonl)
    sample_indices = list(range(min(len(retrieved), len(prompts))))
    random.shuffle(sample_indices)
    if args.max_samples is not None:
        sample_indices = sample_indices[: args.max_samples]

    client = AzureOpenAI(
        api_version=args.azure_api_version,
        azure_endpoint=args.azure_endpoint,
        api_key=args.azure_api_key,
    )

    classified = []
    for idx in tqdm(sample_indices):
        relevant, distracting, neutral, predictions = [], [], [], []
        for doc in retrieved[idx]["ctxs"][: args.max_docs]:
            prediction = classify_document(client, args.classifier_model, prompts[idx], answers[idx], doc_text(doc))
            predictions.append(prediction)
            label = prediction.get("label")
            if label == "relevant":
                relevant.append(doc)
            elif label == "distracting":
                distracting.append(doc)
            else:
                neutral.append(doc)
        classified.append(
            {
                "idx": idx,
                "relevant": relevant,
                "distracting": distracting,
                "neutral": neutral,
                "prediction": predictions,
            }
        )

    random_pool = [doc for row in classified for doc in (row["neutral"] or row["distracting"] or row["relevant"])]
    retrieval_doc = []
    for row in classified:
        other_docs = [doc for doc in random_pool if doc not in row["relevant"] and doc not in row["distracting"]]
        random.shuffle(other_docs)
        retrieval_doc.append(
            {
                "idx": row["idx"],
                "relevant": row["relevant"],
                "distracting": row["distracting"],
                "random": other_docs[: args.max_docs],
            }
        )

    split_dir = args.root / "split_doc" / args.dataset
    write_json(split_dir / "classified.json", classified)
    write_json(args.root / "retrieval_doc" / f"{args.dataset}.json", retrieval_doc)
    print(f"Saved document labels to {split_dir} and retrieval doc sets for {args.dataset}")


if __name__ == "__main__":
    main()


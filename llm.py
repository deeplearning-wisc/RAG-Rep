"""Model and prompting helpers used by pipeline scripts."""

from __future__ import annotations

import gc
import json
from typing import Any

import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig, CompilationLevel
from vllm.distributed.parallel_state import destroy_model_parallel

from understanding_rag.config import JUDGE_MODEL, MODELS
from understanding_rag.data import JUDGE_PROMPT

GENERATION_PARAMS = SamplingParams(temperature=0, top_p=1, max_tokens=256, n=1)
JUDGE_PARAMS = SamplingParams(temperature=0, top_p=1, max_tokens=20, n=1)


def is_base_model(model_name: str) -> bool:
    return model_name.endswith("-base")


def make_llm(model_id: str, tensor_parallel_size: int, max_model_len: int) -> LLM:
    return LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        compilation_config=CompilationConfig(
            level=CompilationLevel.PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        ),
    )


def unload_llm(llm: LLM) -> None:
    destroy_model_parallel()
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def generate_texts(llm: LLM, model_name: str, prompts: list[str]) -> list[str]:
    if is_base_model(model_name):
        outputs = llm.generate(prompts, GENERATION_PARAMS, use_tqdm=True)
    else:
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        outputs = llm.chat(messages, GENERATION_PARAMS, use_tqdm=True)
    return [output.outputs[0].text.strip().split("\n\n", 1)[0].split("\n", 1)[0] for output in outputs]


def judge_answers(
    rows: list[dict[str, Any]],
    tensor_parallel_size: int,
    max_model_len: int = 4096,
    judge_model: str = JUDGE_MODEL,
) -> list[dict[str, Any]]:
    llm = make_llm(judge_model, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)
    try:
        judge_prompts = [
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        question=row["query"],
                        ground_truths="\n".join(f"- {gt}" for gt in row["ground_truth"]),
                        model_output=row["generated_text"],
                    ),
                }
            ]
            for row in rows
        ]
        outputs = llm.chat(judge_prompts, JUDGE_PARAMS, use_tqdm=True)
    finally:
        unload_llm(llm)

    judged = []
    for row, output in zip(rows, outputs, strict=True):
        text = output.outputs[0].text.strip()
        try:
            verdict = json.loads(text)["verdict"]
        except (json.JSONDecodeError, KeyError):
            verdict = "incorrect"
        judged.append({**row, "prediction": verdict})
    return judged


def model_id(model_name: str) -> str:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model {model_name!r}. Expected one of: {sorted(MODELS)}")
    return MODELS[model_name]


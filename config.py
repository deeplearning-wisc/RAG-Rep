"""Shared constants for the main Understanding RAG pipeline."""

from __future__ import annotations

DATASETS = ("trivia", "nq", "pop", "strategy")

INSTRUCTION_MODELS = {
    "gemma3-27B": "google/gemma-3-27b-it",
    "llama4-17B": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "qwen3-80B": "Qwen/Qwen3-Next-80B-A3B-Instruct",
}

BASE_MODELS = {
    "gemma3-27B-base": "google/gemma-3-27b-pt",
    "llama4-17B-base": "meta-llama/Llama-4-Scout-17B-16E",
    "qwen3-80B-base": "Qwen/Qwen3-Next-80B-A3B",
}

MODELS = {**INSTRUCTION_MODELS, **BASE_MODELS}
BASE_MODEL_TO_INSTRUCTION_TOKENIZER = {
    "gemma3-27B-base": "gemma3-27B",
    "llama4-17B-base": "llama4-17B",
    "qwen3-80B-base": "qwen3-80B",
}

JUDGE_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
MAX_RETRIEVED_DOCS = 20
RANDOM_SEED = 42

SINGLE_DOC_SETTINGS = ("relevant", "distracting", "random")
MULTI_DOC_SETTINGS = ("relevant_3_distracting", "relevant_3_random", "all_20")
RETRIEVAL_SETTINGS = SINGLE_DOC_SETTINGS + MULTI_DOC_SETTINGS

PLOT_COLORS = {
    "relevant": "#2A9D8F",
    "distracting": "#E76F51",
    "random": "#E9C46A",
    "no_doc": "#BBBBBB",
    "relevant_3_distracting": "#E76F51",
    "relevant_3_random": "#E9C46A",
    "all_20": "#457B9D",
}

LAYER_SLICES = {
    "gemma3-27B": (12, 26, 30, 60),
    "gemma3-27B-base": (12, 26, 30, 60),
    "llama4-17B": (12, 23, 35, 46),
    "llama4-17B-base": (12, 23, 35, 46),
    "qwen3-80B": (12, 19, 31, 46),
    "qwen3-80B-base": (12, 19, 31, 46),
}


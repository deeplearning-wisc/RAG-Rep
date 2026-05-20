# How Retrieved Context Shapes Internal Representations in RAG

By [Samuel Yeh](https://mhyeh.github.io/) and [Sharon Li](https://pages.cs.wisc.edu/~sharonli/index.html).

[![Paper](https://img.shields.io/badge/arXiv-2602.20091-orange)](https://arxiv.org/abs/2602.20091)


This codebase is the official implementation for the paper, "How Retrieved Context Shapes Internal Representations in RAG." It covers the main experiment pipeline:

1. Generate question-only answers and split them into `correct` / `incorrect`.
2. Export questions for retrieval in a separate project.
3. Categorize retrieved documents as `relevant`, `distracting`, or `random`.
4. Generate and judge answers for retrieval settings:
   `relevant`, `distracting`, `random`, `relevant_3_distracting`, `relevant_3_random`, and `all_20`.
5. Extract last prompt-token representations for each setting.
6. Plot single-doc last-layer PCA.
7. Plot multi-doc last-layer PCA.
8. Plot single-doc PCA across layers.
9. Plot cosine similarity versus response correctness.

Supported datasets are [trivia](https://nlp.cs.washington.edu/triviaqa/), [nq](https://github.com/google-research-datasets/natural-questions), [pop](https://huggingface.co/datasets/akariasai/PopQA), and [strategy](https://huggingface.co/datasets/ChilleD/StrategyQA).

Supported models are `gemma3-27B`, `llama4-17B`, `qwen3-80B`, plus their `-base` variants.

## Example

Run commands from the repository root.

```bash
python -m understanding_rag.generate_question_only --dataset trivia --model gemma3-27B --root .
python -m understanding_rag.prepare_retrieval_data --dataset trivia --root .
```

We use the MassiveDS retrieval-scaling pipeline to retrieve documents. After exporting queries with `prepare_retrieval_data`, run retrieval with [RulinShao/retrieval-scaling](https://github.com/RulinShao/retrieval-scaling); see that repository for setup and execution details. The downstream scripts expect the retrieval output as a JSONL file with `ctxs`.

After retrieval writes the JSONL file, categorize documents:

```bash
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_KEY="..."
python -m understanding_rag.categorize_retrieved_documents \
  --dataset trivia \
  --root . \
  --retrieved-jsonl retrieved_results/post_processed/dedup_merged_trivia_top1000.jsonl
```

Generate retrieval answers, judge them, and extract representations:

```bash
python -m understanding_rag.generate_with_retrieval \
  --dataset trivia \
  --model gemma3-27B \
  --root . \
  --retrieved-jsonl retrieved_results/post_processed/dedup_merged_trivia_top1000.jsonl

python -m understanding_rag.extract_prompt_representations \
  --dataset trivia \
  --model gemma3-27B \
  --difficulty correct \
  --root .
```

Plot scripts are under `understanding_rag/plot_scripts`, or use the shared CLI:

```bash
python -m understanding_rag.plotting single-last --root .
python -m understanding_rag.plotting multiple-last --root .
python -m understanding_rag.plotting single-diff-layer --root .
python -m understanding_rag.plotting sim-vs-response --root .
```


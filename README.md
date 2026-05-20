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
python generate_question_only.py --dataset trivia --model gemma3-27B --root .
python prepare_retrieval_data.py --dataset trivia --root .
```

We use the MassiveDS retrieval-scaling pipeline to retrieve documents. After exporting queries with `prepare_retrieval_data`, run retrieval with [RulinShao/retrieval-scaling](https://github.com/RulinShao/retrieval-scaling); see that repository for setup and execution details. The downstream scripts expect the retrieval output as a JSONL file with `ctxs`.

After retrieval writes the JSONL file, categorize documents:

```bash
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_KEY="..."
python categorize_retrieved_documents.py \
  --dataset trivia \
  --root . \
  --retrieved-jsonl retrieved_results/post_processed/dedup_merged_trivia_top1000.jsonl
```

Generate retrieval answers, judge them, and extract representations:

```bash
python generate_with_retrieval.py \
  --dataset trivia \
  --model gemma3-27B \
  --root . \
  --retrieved-jsonl retrieved_results/post_processed/dedup_merged_trivia_top1000.jsonl

python extract_prompt_representations.py \
  --dataset trivia \
  --model gemma3-27B \
  --difficulty correct \
  --root .
```

Plot scripts are under `plot_scripts`, or use the shared CLI:

```bash
python plotting.py single-last --root .
python plotting.py multiple-last --root .
python plotting.py single-diff-layer --root .
python plotting.py sim-vs-response --root .
```


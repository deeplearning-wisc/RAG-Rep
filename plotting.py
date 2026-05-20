"""Plot PCA and cosine-similarity figures for the canonical pipeline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from understanding_rag.config import DATASETS, INSTRUCTION_MODELS, BASE_MODELS, LAYER_SLICES, PLOT_COLORS

DIFFICULTIES = ("correct", "incorrect")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plot", choices=("single-last", "multiple-last", "single-diff-layer", "sim-vs-response"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--models", nargs="+", default=None)
    return parser.parse_args()


def load_unique_reps(root: Path, dataset: str, model: str, difficulty: str, setting: str) -> list[dict]:
    reps = torch.load(root / "prompt_rep" / dataset / model / difficulty / f"{setting}.pt", map_location="cpu")
    seen, rows = set(), []
    for row in reps:
        if row["idx"] in seen:
            continue
        seen.add(row["idx"])
        rows.append(row)
    return rows


def default_models(base: bool) -> list[str]:
    return list(BASE_MODELS if base else INSTRUCTION_MODELS)


def pca_points(root: Path, dataset: str, model: str, settings: list[str], layer: int = -2):
    vectors, labels, predictions = [], [], []
    for difficulty in DIFFICULTIES:
        for setting in settings:
            rows = load_unique_reps(root, dataset, model, difficulty, setting)
            vectors.extend(row["rep"][layer].float().numpy() for row in rows)
            labels.extend([setting] * len(rows))
            predictions.extend(row.get("prediction", difficulty) for row in rows)
    zipped = list(zip(vectors, labels, predictions))
    random.shuffle(zipped)
    vectors, labels, predictions = zip(*zipped)
    return PCA(n_components=2).fit_transform(vectors), labels, predictions


def plot_single_last(root: Path, output_dir: Path, datasets: list[str], models: list[str], base: bool) -> None:
    settings = ["relevant", "distracting", "random", "no_doc"]
    fig, axes = plt.subplots(len(models), len(datasets), figsize=(5 * len(datasets), 5 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            xy, labels, _ = pca_points(root, dataset, model, settings)
            sns.scatterplot(
                x=xy[:, 0],
                y=xy[:, 1],
                hue=labels,
                palette=[PLOT_COLORS[s] for s in settings],
                hue_order=settings,
                s=30,
                alpha=0.7,
                ax=axes[i, j],
                legend=False,
            )
            axes[i, j].set_title(f"{dataset} / {model}")
            axes[i, j].set(xlabel="", ylabel="")
    suffix = "_base" if base else ""
    save(fig, output_dir / f"single_last_layer_pca{suffix}")


def plot_multiple_last(root: Path, output_dir: Path, datasets: list[str], models: list[str]) -> None:
    settings = ["relevant", "relevant_3_distracting", "relevant_3_random", "all_20"]
    fig, axes = plt.subplots(len(models), len(datasets), figsize=(5 * len(datasets), 5 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        for j, dataset in enumerate(datasets):
            xy, labels, _ = pca_points(root, dataset, model, settings)
            sns.scatterplot(
                x=xy[:, 0],
                y=xy[:, 1],
                hue=labels,
                palette=[PLOT_COLORS[s] for s in settings],
                hue_order=settings,
                s=30,
                alpha=0.7,
                ax=axes[i, j],
                legend=False,
            )
            axes[i, j].set_title(f"{dataset} / {model}")
            axes[i, j].set(xlabel="", ylabel="")
    save(fig, output_dir / "multiple_last_layer_pca")


def plot_diff_layer(root: Path, output_dir: Path, datasets: list[str], models: list[str], base: bool) -> None:
    settings = ["relevant", "distracting", "random", "no_doc"]
    for model in models:
        layers = LAYER_SLICES[model]
        fig, axes = plt.subplots(len(datasets), len(layers), figsize=(5 * len(layers), 5 * len(datasets)), squeeze=False)
        for i, dataset in enumerate(datasets):
            for j, layer in enumerate(layers):
                xy, labels, _ = pca_points(root, dataset, model, settings, layer=layer)
                sns.scatterplot(
                    x=xy[:, 0],
                    y=xy[:, 1],
                    hue=labels,
                    palette=[PLOT_COLORS[s] for s in settings],
                    hue_order=settings,
                    s=30,
                    alpha=0.7,
                    ax=axes[i, j],
                    legend=False,
                )
                axes[i, j].set_title(f"{dataset} layer {layer}")
                axes[i, j].set(xlabel="", ylabel="")
        save(fig, output_dir / f"single_diff_layer_pca_{model}")


def plot_sim_vs_response(root: Path, output_dir: Path, datasets: list[str], models: list[str], base: bool) -> None:
    settings = ["relevant", "distracting", "random"]
    for model in models:
        fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 2.5 * len(datasets)), sharex=True, squeeze=False)
        for i, dataset in enumerate(datasets):
            sims, labels, predictions = [], [], []
            no_doc = {}
            for difficulty in DIFFICULTIES:
                for row in load_unique_reps(root, dataset, model, difficulty, "no_doc"):
                    no_doc[row["idx"]] = row
                for setting in settings:
                    rows = load_unique_reps(root, dataset, model, difficulty, setting)
                    for row in rows:
                        if row["idx"] not in no_doc:
                            continue
                        sims.append(F.cosine_similarity(no_doc[row["idx"]]["rep"][-2], row["rep"][-2], dim=0).item())
                        labels.append(setting)
                        predictions.append(row.get("prediction", difficulty))
            sns.stripplot(
                x=sims,
                y=labels,
                hue=predictions,
                hue_order=["correct", "incorrect", "abstain"],
                palette=["#2A9D8F", "#E76F51", "#BBBBBB"],
                alpha=0.7,
                s=8,
                jitter=0.18,
                ax=axes[i, 0],
                legend=False,
            )
            axes[i, 0].set_title(dataset)
            axes[i, 0].set(xlabel="", ylabel="")
            axes[i, 0].set_xlim(0.5 if base else 0, 1)
        suffix = "_base" if base else ""
        save(fig, output_dir / f"sim_vs_response_{model}{suffix}")


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{stem}.png", bbox_inches="tight")
    fig.savefig(f"{stem}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    sns.set_style("whitegrid", {"xtick.direction": "out", "ytick.direction": "out"})
    output_dir = args.output_dir or args.root / "images"
    models = args.models or default_models(args.base)
    if args.plot == "single-last":
        plot_single_last(args.root, output_dir, args.datasets, models, args.base)
    elif args.plot == "multiple-last":
        plot_multiple_last(args.root, output_dir, args.datasets, models)
    elif args.plot == "single-diff-layer":
        plot_diff_layer(args.root, output_dir, args.datasets, models, args.base)
    elif args.plot == "sim-vs-response":
        plot_sim_vs_response(args.root, output_dir, args.datasets, models, args.base)


if __name__ == "__main__":
    main()


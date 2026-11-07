"""Plot the matched EXP-172 evaluation from the archived summary."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PAPER = Path(__file__).resolve().parent
SOURCE = PAPER.parent / "results" / "EXP-172-received-summary.json"
PANELS = [
    ("edu_1024", "FineWeb-Edu"),
    ("english_stories_1024", "TinyStories"),
    ("russian_wikipedia_1024", "Russian Wikipedia"),
]
TASKS = [
    ("arc_easy", "ARC-Easy", "acc_norm"),
    ("arc_challenge", "ARC-Challenge", "acc_norm"),
    ("piqa", "PIQA", "acc_norm"),
    ("winogrande", "WinoGrande", "acc"),
    ("hellaswag", "HellaSwag", "acc_norm"),
    ("mmlu", "MMLU", "acc"),
]
MODELS = {
    "teacher": ("Original Qwen", "#485363", "D"),
    "gqa4": ("4 attention · 24 Mamba", "#0072B2", "o"),
    "gqa7": ("7 attention · 21 Mamba", "#B14F09", "s"),
}


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    summary = data["summary"]
    if data["status"] != "completed" or data["contract"]["source_step"] != 5120:
        raise ValueError("Expected the complete evaluation of frozen EXP-171 CURRENT5120")

    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, (nll, accuracy) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    fig.suptitle("Quality after Qwen → Mamba-3 MIMO transfer", fontweight="bold")
    for ax in (nll, accuracy):
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#e3e7ed")
        ax.set_axisbelow(True)

    for index, (panel, _) in enumerate(PANELS):
        for offset, (role, (_, color, marker)) in zip((0.20, 0, -0.20), MODELS.items()):
            point = summary["nll"][role][panel]
            if (point["n_windows"], point["targets"]) != (32, 32736):
                raise ValueError(f"Unexpected coverage for {role}/{panel}")
            nll.scatter(point["nll"], index + offset, color=color, marker=marker, s=45, zorder=3)
    nll.set(yticks=range(len(PANELS)), yticklabels=[label for _, label in PANELS],
            xlabel="NLL (lower is better)", xlim=(1.2, 2.7), ylim=(2.55, -0.55))
    nll.set_title("A   Text prediction", loc="left", fontweight="bold")

    accuracy.axvline(0, color=MODELS["teacher"][1], linestyle="--", linewidth=1.3)
    for index, (task, _, metric) in enumerate(TASKS):
        for offset, role in ((-0.15, "gqa4"), (0.15, "gqa7")):
            _, color, marker = MODELS[role]
            paired = summary["paired"][role][task]
            if paired["n_common"] != 1000:
                raise ValueError(f"Unexpected paired coverage for {role}/{task}")
            point = paired[metric]
            delta = 100 * point["hybrid_minus_teacher"]
            lo, hi = (100 * bound for bound in point["paired_bootstrap_ci95"])
            accuracy.errorbar(delta, index + offset, xerr=[[delta - lo], [hi - delta]],
                              fmt=marker, color=color, markersize=6, capsize=3)
    accuracy.set(yticks=range(len(TASKS)), yticklabels=[label for _, label, _ in TASKS],
                 xlabel="Accuracy difference vs. Qwen (percentage points)",
                 xlim=(-23, 12), ylim=(5.55, -0.55))
    accuracy.set_title("B   Matched zero-shot tasks", loc="left", fontweight="bold")

    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", label=label)
               for label, color, marker in MODELS.values()]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)
    for extension in ("png", "pdf", "svg"):
        output = PAPER / "figures" / f"clean_transfer_quality.{extension}"
        fig.savefig(output, dpi=200)
        if extension == "svg":
            lines = output.read_text(encoding="utf-8").splitlines()
            output.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()

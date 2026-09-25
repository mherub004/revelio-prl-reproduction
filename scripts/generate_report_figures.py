#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path.home() / "revelio" / "report_artifacts"
LOG_DIR = BASE_DIR / "logs"
FIGURE_DIR = BASE_DIR / "figures"
TABLE_DIR = BASE_DIR / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

EPOCH_PATTERN = re.compile(
    r"Epoch\s+(\d+):\s+Test Loss:\s+([0-9.]+),\s+"
    r"Test Accuracy:\s+([0-9.]+)"
)


def newest_matching(pattern: str) -> Path:
    files = list(LOG_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No log matching {pattern!r} was found in {LOG_DIR}"
        )

    return max(files, key=lambda path: path.stat().st_mtime)


def exact_or_matching(filename: str, fallback_pattern: str) -> Path:
    exact_path = LOG_DIR / filename

    if exact_path.exists():
        return exact_path

    return newest_matching(fallback_pattern)


def parse_log(log_path: Path) -> list[dict[str, float | int]]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    records: list[dict[str, float | int]] = []

    for match in EPOCH_PATTERN.finditer(text):
        records.append(
            {
                "epoch": int(match.group(1)),
                "test_loss": float(match.group(2)),
                "test_accuracy": float(match.group(3)) * 100.0,
            }
        )

    if not records:
        raise RuntimeError(
            f"No epoch results were found in {log_path}"
        )

    return records


# ------------------------------------------------------------
# Select the completed experiment logs
# ------------------------------------------------------------

layer_logs = {
    "bottleneck": newest_matching(
        "v100_sd15_bottleneck0_t25_*.err"
    ),
    "up_ft0": newest_matching(
        "v100_sd15_upft0_t25_*.err"
    ),
    # Filename says t50, but the saved result path confirms t=25.
    "up_ft1": exact_or_matching(
        "sd15_upft1_t50_1671566.err",
        "*upft1*t50*1671566*.err",
    ),
    "up_ft2": exact_or_matching(
        "sd15_upft2_t25_1668726.err",
        "*upft2*t25*.err",
    ),
}

timestep_logs = {
    25: layer_logs["up_ft1"],
    50: exact_or_matching(
        "sd15_upft1_t50_1668729.err",
        "*upft1*t50*1668729*.err",
    ),
    75: exact_or_matching(
        "sd15_upft1_t75_1668730.err",
        "*upft1*t75*.err",
    ),
    100: exact_or_matching(
        "sd15_upft1_t100_from_t75_1693487.err",
        "*upft1*t100*1693487*.err",
    ),
}

layer_records = {
    layer: parse_log(log_path)
    for layer, log_path in layer_logs.items()
}

timestep_records = {
    timestep: parse_log(log_path)
    for timestep, log_path in timestep_logs.items()
}

layer_final = {
    layer: records[-1]["test_accuracy"]
    for layer, records in layer_records.items()
}

timestep_final = {
    timestep: records[-1]["test_accuracy"]
    for timestep, records in timestep_records.items()
}

# Approximate reference values supplied in the project proposal.
paper_layer_reference = {
    "bottleneck": 70.0,
    "up_ft0": 85.0,
    "up_ft1": 88.61,
    "up_ft2": 65.0,
}

# ------------------------------------------------------------
# Figure 1: layer comparison
# ------------------------------------------------------------

layers = ["bottleneck", "up_ft0", "up_ft1", "up_ft2"]
x_positions = list(range(len(layers)))
bar_width = 0.36

fig, ax = plt.subplots(figsize=(8.0, 4.8))

ax.bar(
    [x - bar_width / 2 for x in x_positions],
    [paper_layer_reference[layer] for layer in layers],
    width=bar_width,
    label="Paper reference",
)

ax.bar(
    [x + bar_width / 2 for x in x_positions],
    [layer_final[layer] for layer in layers],
    width=bar_width,
    label="This reproduction",
)

ax.set_xlabel("Stable Diffusion 1.5 layer")
ax.set_ylabel("Top-1 test accuracy (%)")
ax.set_title("Diff-C accuracy across SD-1.5 layers at t=25")
ax.set_xticks(x_positions)
ax.set_xticklabels(
    ["bottleneck", "up_ft0", "up_ft1", "up_ft2"]
)
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "layer_comparison.pdf",
    bbox_inches="tight",
)
fig.savefig(
    FIGURE_DIR / "layer_comparison.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ------------------------------------------------------------
# Figure 2: timestep sweep
# ------------------------------------------------------------

timesteps = [25, 50, 75, 100]

fig, ax = plt.subplots(figsize=(7.5, 4.6))

ax.plot(
    timesteps,
    [timestep_final[timestep] for timestep in timesteps],
    marker="o",
    linewidth=2,
    label="This experiment",
)

ax.axhline(
    88.61,
    linestyle="--",
    linewidth=1.5,
    label="Paper result at t=25 (88.61%)",
)

for timestep in timesteps:
    accuracy = timestep_final[timestep]

    ax.annotate(
        f"{accuracy:.2f}%",
        (timestep, accuracy),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
    )

ax.set_xlabel("Diffusion timestep")
ax.set_ylabel("Top-1 test accuracy (%)")
ax.set_title("Effect of diffusion timestep using SD-1.5 up_ft1")
ax.set_xticks(timesteps)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "timestep_sweep.pdf",
    bbox_inches="tight",
)
fig.savefig(
    FIGURE_DIR / "timestep_sweep.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ------------------------------------------------------------
# Figure 3: training curves for the timestep experiment
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.0, 4.8))

for timestep in timesteps:
    records = timestep_records[timestep]

    ax.plot(
        [record["epoch"] for record in records],
        [record["test_accuracy"] for record in records],
        linewidth=1.8,
        label=f"t={timestep}",
    )

ax.set_xlabel("Epoch")
ax.set_ylabel("Top-1 test accuracy (%)")
ax.set_title("Diff-C test accuracy during training")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "training_curves.pdf",
    bbox_inches="tight",
)
fig.savefig(
    FIGURE_DIR / "training_curves.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# ------------------------------------------------------------
# CSV and LaTeX table: layer results
# ------------------------------------------------------------

layer_csv = TABLE_DIR / "layer_comparison.csv"

with layer_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Layer",
            "Paper accuracy (%)",
            "Reproduced accuracy (%)",
            "Difference (pp)",
        ]
    )

    for layer in layers:
        paper = paper_layer_reference[layer]
        reproduced = layer_final[layer]

        writer.writerow(
            [
                layer,
                f"{paper:.2f}",
                f"{reproduced:.2f}",
                f"{reproduced - paper:+.2f}",
            ]
        )

layer_tex = TABLE_DIR / "layer_comparison.tex"

with layer_tex.open("w", encoding="utf-8") as file:
    file.write("\\begin{tabular}{lrrr}\n")
    file.write("\\toprule\n")
    file.write(
        "Layer & Paper (\\%) & This work (\\%) "
        "& Difference (pp) \\\\\n"
    )
    file.write("\\midrule\n")

    for layer in layers:
        paper = paper_layer_reference[layer]
        reproduced = layer_final[layer]

        latex_layer = layer.replace("_", "\\_")

        file.write(
            f"{latex_layer} & {paper:.2f} & "
            f"{reproduced:.2f} & "
            f"{reproduced - paper:+.2f} \\\\\n"
        )

    file.write("\\bottomrule\n")
    file.write("\\end{tabular}\n")

# ------------------------------------------------------------
# CSV and LaTeX table: timestep results
# ------------------------------------------------------------

timestep_csv = TABLE_DIR / "timestep_results.csv"

with timestep_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Timestep", "Final test accuracy (%)", "Final test loss"]
    )

    for timestep in timesteps:
        final_record = timestep_records[timestep][-1]

        writer.writerow(
            [
                timestep,
                f"{final_record['test_accuracy']:.3f}",
                f"{final_record['test_loss']:.5f}",
            ]
        )

timestep_tex = TABLE_DIR / "timestep_results.tex"

with timestep_tex.open("w", encoding="utf-8") as file:
    file.write("\\begin{tabular}{rrr}\n")
    file.write("\\toprule\n")
    file.write(
        "Timestep & Test accuracy (\\%) & Test loss \\\\\n"
    )
    file.write("\\midrule\n")

    for timestep in timesteps:
        final_record = timestep_records[timestep][-1]

        file.write(
            f"{timestep} & "
            f"{final_record['test_accuracy']:.3f} & "
            f"{final_record['test_loss']:.5f} \\\\\n"
        )

    file.write("\\bottomrule\n")
    file.write("\\end{tabular}\n")

print("\nGenerated figures:")
for path in sorted(FIGURE_DIR.glob("*")):
    print(f"  {path}")

print("\nLayer final results:")
for layer in layers:
    print(
        f"  {layer:12s}: {layer_final[layer]:.3f}%"
    )

print("\nTimestep final results:")
for timestep in timesteps:
    print(
        f"  t={timestep:<3d}: "
        f"{timestep_final[timestep]:.3f}%"
    )

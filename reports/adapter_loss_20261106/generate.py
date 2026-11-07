"""CPU-only, reproducible plots of recorded Unsloth training/evaluation loss."""
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/soomac-loss-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LOCAL = Path("/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_unsloth_local_authorized_20261101")
SOURCES = {
    "Decision": ROOT / "training/learning_first_v43/final/goals/GL10/dqp_adoption_20261105/archive/extracted/results/checkpoints/attempt-20a5090493204fe58601d6016dd47a45/decision/checkpoint-1638/trainer_state.json",
    "Recommendation": LOCAL / "recommendation/attempt-9726a651729b41a6bd73d0b2a0bea084/recommendation/checkpoint-234/trainer_state.json",
    "Response": LOCAL / "response/attempt-63872c5d7d584aab81aa73bbb6cc0155/response/checkpoint-624/trainer_state.json",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "normal",
    "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": True, "axes.spines.right": True,
    "axes.edgecolor": "black",
    "svg.fonttype": "none", "savefig.facecolor": "white",
})


def points(history, key):
    result = []
    seen = set()
    for row in history:
        if key not in row:
            continue
        point = {"step": row["step"], "epoch": row["epoch"], "loss": row[key]}
        identity = (point["step"], point["epoch"], point["loss"])
        if identity in seen:
            continue  # Identical repeated final evaluation; keep one plotted point.
        seen.add(identity)
        assert 0 < point["epoch"] <= 6 and point["loss"] >= 0
        result.append(point)
    assert all(a["step"] <= b["step"] for a, b in zip(result, result[1:]))
    return result


def draw(ax, role, data):
    for key, label, color, marker in [
        ("train", "Train", "tab:blue", None),
    ]:
        rows = data[key]
        assert all(r["loss"] > 0 for r in rows), "Log axis requires positive losses"
        ax.plot([r["epoch"] for r in rows], [r["loss"] for r in rows],
                label=label, color=color, lw=1.2, marker=marker, markersize=3)
    values = [r["loss"] for r in data["train"]]
    low, high = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
    ax.set(title=f"{role} Adapter Loss", xlabel="Epoch", ylabel="Loss", xlim=(0, 6))
    ax.set_yscale("log")
    ax.set_ylim(10**low, 10**high)
    ax.set_yticks([10**power for power in range(low, high + 1)])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks(range(7))
    ax.grid(axis="y", which="major", color="#DCE2E8", lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="best", frameon=False, fontsize=9)


evidence = {
    "notes": [
        "Recorded log_history epoch coordinates; no smoothing or fabricated epoch-zero point.",
        "Training Loss is the Trainer logged interval loss, not an epoch mean.",
        "Validation Loss is the historical training-time evaluation, not scenario accuracy or a newly audited evaluation.",
        "Figures and epoch table contain training loss ONLY. Validation records are retained only as source evidence; original files are unchanged.",
        "All runs trained for six epochs. Runtime selection: Decision step1638, Recommendation step234, Response step150 (epoch1.4423076923).",
        "One panel per adapter, logarithmic loss axis with four-decimal tick labels. Full observed loss range is retained.",
        "Training curves end at their last logged training point; source values are unchanged.",
    ],
    "adapters": {},
}
for role, source in SOURCES.items():
    raw = source.read_bytes()
    state = json.loads(raw)
    assert state["epoch"] == 6.0
    data = {"source": str(source), "sha256": hashlib.sha256(raw).hexdigest(),
            "final_step": state["global_step"], "train": points(state["log_history"], "loss"),
            "validation": points(state["log_history"], "eval_loss")}
    evidence["adapters"][role] = data
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    draw(ax, role, data)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{role.lower()}_adapter_loss.{extension}", dpi=300)
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(17.4, 4.6), layout="constrained")
for i, (role, data) in enumerate(evidence["adapters"].items()):
    draw(axes[i], role, data)
fig.savefig(OUT / "all_adapter_losses.png", dpi=200)
plt.close(fig)
(OUT / "loss_data_and_sources.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({role: {"training_points": len(d["train"]), "validation_points": len(d["validation"])}
                  for role, d in evidence["adapters"].items()}))

summary = []
for epoch in range(1, 7):
    row = {"epoch": epoch}
    for role, data in evidence["adapters"].items():
        train = [r for r in data["train"] if epoch - 1 < r["epoch"] <= epoch][-1]
        row[role] = {"train_last_logged": train}
    summary.append(row)
(OUT / "epoch_loss_table.json").write_text(json.dumps({
    "definition": "Train ONLY: last logged training interval within each epoch, NOT epoch-average loss.",
    "rows": summary}, ensure_ascii=False, indent=2) + "\n")
for row in summary:
    print(" | ".join([str(row["epoch"])] + [f"{row[role]['train_last_logged']['loss']:.4f}" for role in SOURCES]))

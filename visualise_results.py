"""Generate result figures for the deepfake-detector evaluation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CSV_FILEPATH = "/Users/annastevens/Library/Mobile Documents/com~apple~CloudDocs/"\
               + "Documents/2026-Sem2/CYBR7902/Initial results/"\
               + "final_results_1789616371.2015939.csv"

df = pd.read_csv(CSV_FILEPATH)


def add_balanced_accuracy(r):
    """Compute balanced accuracy and real-frame count for one results row.

    The number of fake (positive) frames ``P`` is recovered from the row's
    summary metrics, since the total errors satisfy
    ``n * (1 - accuracy) = P * (recall / precision + 1 - 2 * recall)``.
    From ``P`` the confusion matrix, true negative rate and balanced
    accuracy ``(recall + TNR) / 2`` follow.

    Args:
        r: A row with ``n``, ``accuracy``, ``precision`` and ``recall``.

    Returns:
        A Series with ``balanced_acc`` and ``n_real``. Perfect
        classification gives a balanced accuracy of 1.0 and an unknown
        (NaN) real count. Rows with zero precision or recall, or where the
        derivation is undefined, give NaN for both values.
    """
    n, acc, prec, rec = r["n"], r["accuracy"], r["precision"], r["recall"]
    if acc == 1.0:  # Perfect classification: every frame correct
        return pd.Series({"balanced_acc": 1.0, "n_real": float("nan")})
    denom = rec / prec + 1 - 2 * rec if prec > 0 else 0
    if prec == 0 or rec == 0 or denom == 0:
        return pd.Series({"balanced_acc": float("nan"),
                          "n_real": float("nan")})
    P = n * (1 - acc) / denom  # Fake frames
    TP = rec * P
    TN = acc * n - TP
    tnr = min(max(TN / (n - P), 0), 1)
    return pd.Series({"balanced_acc": (rec + tnr) / 2,
                      "n_real": n - P})


df[["balanced_acc", "n_real"]] = df.apply(add_balanced_accuracy, axis=1)


# Models, techniques and display names used throughout the figures
models = ["xception", "efficientnet7", "swin"]
model_names = {"xception": "XceptionNet", "efficientnet7": "EfficientNet-B7",
               "swin": "Swin Transformer"}
techs = ["clean", "compression", "blur", "resize", "noise"]
tech_names = [t.capitalize() for t in techs]

gender_df = df[df.gender.isin(["men", "women"]) & (df.race == "all")
               & df.model.isin(models)]
table = gender_df.pivot_table(index=["model", "gender"], columns="technique",
                              values="balanced_acc")
table = table.reindex(
    pd.MultiIndex.from_product([models, ["men", "women"]])
)[techs]


# ---------- Gender gap (women - men) ----------
# Aligned by model/technique
diff = table.xs("women", level=1) - table.xs("men", level=1)
diff.index = [model_names[m] for m in diff.index]
diff.columns = tech_names

plt.figure(figsize=(7, 4))
sns.heatmap(diff, annot=True, fmt=".2f", cmap="RdBu", center=0,
            vmin=-0.15, vmax=0.15, cbar_kws={"label": "Balanced accuracy"})
plt.title("Gender gap in balanced accuracy")
plt.tight_layout()
plt.savefig("image_results/heatmap_gender_diff.png", dpi=200,
            bbox_inches="tight")


# ---------- Model x race x gender, one panel per model ----------
races = ["African", "Asian (East)", "Asian (South)",
         "Caucasian (American)", "Caucasian (European)"]
row_index = pd.MultiIndex.from_tuples(
    [(r, g) for r in races for g in ["men", "women"]],
    names=["race", "gender"])

race_df = df[df.gender.isin(["men", "women"]) & (df.race != "all")
             & df.model.isin(models)]

fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True)
for ax, m in zip(axes, models):
    d = race_df[race_df.model == m]
    t = d.pivot_table(index=["race", "gender"], columns="technique",
                      values="balanced_acc").reindex(row_index)[techs]
    nr = d.pivot_table(index=["race", "gender"], columns="technique",
                       values="n_real").reindex(row_index)[techs]
    t.index = [f"{r} |\n{g}" for r, g in t.index]
    t.columns = tech_names

    sns.heatmap(t, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0.0, vmax=1.0, linewidths=0.5, cbar=False,
                annot_kws={"size": 12})
    for y in range(2, len(t), 2):
        ax.axhline(y, color="black", linewidth=1.5)
    ax.set_title(model_names[m], fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12, labelrotation=90)

cbar = fig.colorbar(axes[0].collections[0], ax=axes.tolist(),
                    label="Balanced accuracy", shrink=0.7)
cbar.set_label("Balanced accuracy", fontsize=14)  # Axis title size
cbar.ax.tick_params(labelsize=12)

plt.savefig("image_results/heatmap_model_race_gender.png", dpi=200,
            bbox_inches="tight")


def confusion_from_metrics(r):
    """Reconstruct a confusion matrix from one results row.

    The positive class is "fake". Counts are recovered from ``n``,
    accuracy, precision and recall, then rounded to whole frames.

    Args:
        r: A row with ``n``, ``accuracy``, ``precision`` and ``recall``.

    Returns:
        A 2x2 float array ``[[TN, FP], [FN, TP]]`` of non-negative counts.
    """
    n, acc, prec, rec = r["n"], r["accuracy"], r["precision"], r["recall"]
    denom = rec / prec + 1 - 2 * rec
    P = n * (1 - acc) / denom  # Number of fake frames
    TP = rec * P
    FN = P - TP
    TN = acc * n - TP
    FP = (n - P) - TN
    cm = np.array([[TN, FP], [FN, TP]])
    return np.clip(np.round(cm), 0, None) + 0.0


# ---------- Confusion matrices: EfficientNet B4 vs B7, clean images ----------
cmp_models = {"efficientnet4": "EfficientNet-B4",  # Check key against CSV
              "efficientnet7": "EfficientNet-B7"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (key, name) in zip(axes, cmp_models.items()):
    # Adjust if "all" is coded differently
    row = df[(df.model == key) & (df.technique == "clean")
             & (df.race == "all") & (df.gender == "all")]
    assert len(row) == 1, f"{key}: expected 1 row, got {len(row)}"
    cm = confusion_from_metrics(row.iloc[0])

    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros_like(cm),
                    where=row_sums > 0) * 100
    pct = pct + 0.0
    labels = np.array([[f"{cm[i, j]:,.0f}\n({pct[i, j]:.1f}%)"
                        for j in range(2)]
                       for i in range(2)])

    sns.heatmap(pct, ax=ax, annot=labels, fmt="", cmap="Blues",
                vmin=0, vmax=100, cbar=False, linewidths=0.5,
                annot_kws={"size": 12},
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


plt.tight_layout()
plt.savefig("image_results/confusion_b4_vs_b7_clean.png", dpi=200,
            bbox_inches="tight")


# ---------- Confusion matrices: selected model-technique cases ----------
rows = {
    "EfficientNet-B7 Clean (Working)": df[
        (df.model == "efficientnet7") & (df.technique == "clean")
        & (df.race == "all") & (df.gender == "all")],
    "XceptionNet Blur (All Classified Fake)": df[
        (df.model == "xception") & (df.technique == "blur")
        & (df.race == "all") & (df.gender == "all")],
    "EfficientNet-B7 Blur (Most Classified Real)": df[
        (df.model == "efficientnet7") & (df.technique == "blur")
        & (df.race == "all") & (df.gender == "all")],
    "XceptionNet Resizing (Working)": df[
        (df.model == "xception") & (df.technique == "resize")
        & (df.race == "all") & (df.gender == "all")],
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (key, row) in zip(axes.flatten(), rows.items()):
    assert len(row) == 1, f"{key}: expected 1 row, got {len(row)}"
    cm = confusion_from_metrics(row.iloc[0])

    # Row-normalised (per true class)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{cm[i, j]:,.0f}\n({pct[i, j]:.1f}%)"
                        for j in range(2)]
                       for i in range(2)])

    sns.heatmap(pct, ax=ax, annot=labels, fmt="", cmap="Blues",
                vmin=0, vmax=100, cbar=False, linewidths=0.5,
                annot_kws={"size": 12},
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_title(key)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


plt.tight_layout()
plt.savefig("image_results/confusion_matrices.png", dpi=200,
            bbox_inches="tight")


# ---------- Class ratio vs precision / balanced accuracy ----------
r = pd.read_csv(
    "/Users/annastevens/Library/Mobile Documents/com~apple~CloudDocs/"
    "Documents/2026-Sem2/CYBR7902/Initial results/"
    "final_results_1789616371.2015939.csv"
)
c = r[
    (r.race != "all") & (r.gender != "all") & (r.model != "efficientnet4")
].copy()

# Invert Eq. 4.1-4.4 to recover real/fake frame counts per cell
D = c.recall / c.precision + 1 - 2 * c.recall
c["P"] = (c.n * (1 - c.accuracy) / D).round()  # Fake frames
c["R"] = c.n - c.P  # Real frames
c["ratio"] = c.P / c.R
c["TNR"] = ((c.accuracy * c.n - c.recall * c.P) / c.R).clip(0, 1)
c["BA"] = (c.recall + c.TNR) / 2
collapsed_techniques = ["compression", "blur", "noise"]

c["collapsed"] = (
    c["technique"].isin(collapsed_techniques)
    | ((c["model"] == "swin") & (c["technique"] == "resize"))
)

x = np.logspace(np.log10(c.ratio.min()), np.log10(c.ratio.max()), 100)
fig, ax = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
for a, col, title in zip(ax, ["precision", "BA"],
                         ["Precision", "Balanced accuracy"]):
    for flag, colour, lab in [(True, "tab:red", "collapsed"),
                              (False, "tab:blue", "working")]:
        s = c[c.collapsed == flag]
        a.scatter(s.ratio, s[col], c=colour, alpha=.6, s=18, label=lab)
    if col != "BA":
        a.plot(x, x / (x + 1), "k--", lw=1, label="always-predict-fake")
    a.set_xscale("log")

    a.set_title(title)

ax[1].set_xlabel("fake:real ratio for each ethnicity-gender combination")
ax[0].legend()

plt.tight_layout()
plt.savefig("image_results/ratio_vs_metrics.png", dpi=200)
plt.show()
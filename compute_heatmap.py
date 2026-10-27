import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/annastevens/Library/Mobile Documents/com~apple~CloudDocs/Documents/2026-Sem2/CYBR7902/Initial results/final_results_1789616371.2015939.csv")

def add_balanced_accuracy(r):
    n, acc, prec, rec = r["n"], r["accuracy"], r["precision"], r["recall"]
    if acc == 1.0:   # perfect classification: every frame correct
        return pd.Series({"balanced_acc": 1.0, "n_real": float("nan")})
    denom = rec / prec + 1 - 2 * rec if prec > 0 else 0
    if prec == 0 or rec == 0 or denom == 0:
        return pd.Series({"balanced_acc": float("nan"),
                          "n_real": float("nan")})
    P = n * (1 - acc) / denom               # fake frames
    TP = rec * P
    TN = acc * n - TP
    tnr = min(max(TN / (n - P), 0), 1)
    return pd.Series({"balanced_acc": (rec + tnr) / 2,
                      "n_real": n - P})

df[["balanced_acc", "n_real"]] = df.apply(add_balanced_accuracy, axis=1)

# --- fixed orders and display names (keys must match the CSV exactly) ---
models = ["xception", "efficientnet7", "swin"]
model_names = {"xception": "XceptionNet", "efficientnet7": "EfficientNet-B7",
               "swin": "Swin Transformer"}
techs = ["clean", "compression", "blur", "resize", "noise"]
tech_names = [t.capitalize() for t in techs]

gender_df = df[df.gender.isin(["men", "women"]) & (df.race == "all")
               & df.model.isin(models)]
table = gender_df.pivot_table(index=["model", "gender"], columns="technique",
                              values="balanced_acc")
table = table.reindex(pd.MultiIndex.from_product([models, ["men", "women"]]))[techs]


# ---------- 2) women - men gap ----------
diff = table.xs("women", level=1) - table.xs("men", level=1)   # aligned by model/technique
diff.index = [model_names[m] for m in diff.index]
diff.columns = tech_names

plt.figure(figsize=(7, 4))
sns.heatmap(diff, annot=True, fmt=".2f", cmap="RdBu", center=0, vmin=-0.15, vmax=0.15,
            cbar_kws={"label": "Balanced accuracy"})
plt.title("Gender gap in balanced accuracy")
plt.tight_layout()
plt.savefig("heatmap_gender_diff.png", dpi=200, bbox_inches="tight")
# plt.show()

# ---------- 3) model x race x gender, one panel per model ----------
races = ["African", "Asian (East)", "Asian (South)",
         "Caucasian (American)", "Caucasian (European)"]
row_index = pd.MultiIndex.from_tuples(
    [(r, g) for r in races for g in ["men", "women"]], names=["race", "gender"])

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

    sns.heatmap(t, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0.0, vmax=1.0,
                linewidths=0.5, cbar=False, annot_kws={"size": 12})
    for y in range(2, len(t), 2):
        ax.axhline(y, color="black", linewidth=1.5)
    ax.set_title(model_names[m], fontsize=14)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12, labelrotation=90)

cbar = fig.colorbar(axes[0].collections[0], ax=axes.tolist(),
             label="Balanced accuracy", shrink=0.7)
cbar.set_label("Balanced accuracy", fontsize=14)   # axis title size
cbar.ax.tick_params(labelsize=12)

plt.savefig("heatmap_model_race_gender.png", dpi=200, bbox_inches="tight")
# plt.show()



def confusion_from_metrics(r):
    """Reconstruct [[TN, FP], [FN, TP]] (positive class = fake) from one CSV row."""
    n, acc, prec, rec = r["n"], r["accuracy"], r["precision"], r["recall"]
    denom = rec / prec + 1 - 2 * rec
    P = n * (1 - acc) / denom        # number of fake frames
    TP = rec * P
    FN = P - TP
    TN = acc * n - TP
    FP = (n - P) - TN
    cm = np.array([[TN, FP], [FN, TP]])
    return np.clip(np.round(cm), 0, None) + 0.0

# ---------- 4) confusion matrices: EfficientNet B4 vs B7, clean images ----------
cmp_models = {"efficientnet4": "EfficientNet-B4",   # <-- check key against your CSV
              "efficientnet7": "EfficientNet-B7"}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (key, name) in zip(axes, cmp_models.items()):
    row = df[(df.model == key) & (df.technique == "clean")
             & (df.race == "all") & (df.gender == "all")]   # adjust if "all" is coded differently
    assert len(row) == 1, f"{key}: expected 1 row, got {len(row)}"
    cm = confusion_from_metrics(row.iloc[0])

    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0) * 100
    pct = pct + 0.0
    labels = np.array([[f"{cm[i, j]:,.0f}\n({pct[i, j]:.1f}%)" for j in range(2)]
                       for i in range(2)])

    sns.heatmap(pct, ax=ax, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=100,
                cbar=False, linewidths=0.5, annot_kws={"size": 12},
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


plt.tight_layout()
plt.savefig("confusion_b4_vs_b7_clean.png", dpi=200, bbox_inches="tight")
# plt.show()


# ---------- 5) confusion matrices ----------


rows = {"EfficientNet-B7 Clean (Working)": df[(df.model == "efficientnet7") & (df.technique == "clean")
             & (df.race == "all") & (df.gender == "all")],
        "XceptionNet Blur (All Classified Fake)": df[(df.model == "xception") & (df.technique == "blur")
             & (df.race == "all") & (df.gender == "all")],
        "EfficientNet-B7 Blur (Most Classified Real)": df[(df.model == "efficientnet7") & (df.technique == "blur")
             & (df.race == "all") & (df.gender == "all")],
        "XceptionNet Resizing (Working)": df[(df.model == "xception") & (df.technique == "resize")
             & (df.race == "all") & (df.gender == "all")],}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (key, row) in zip(axes.flatten(), rows.items()):
    assert len(row) == 1, f"{key}: expected 1 row, got {len(row)}"
    cm = confusion_from_metrics(row.iloc[0])

    pct = cm / cm.sum(axis=1, keepdims=True) * 100          # row-normalised (per true class)
    labels = np.array([[f"{cm[i, j]:,.0f}\n({pct[i, j]:.1f}%)" for j in range(2)]
                       for i in range(2)])

    sns.heatmap(pct, ax=ax, annot=labels, fmt="", cmap="Blues", vmin=0, vmax=100,
                cbar=False, linewidths=0.5, annot_kws={"size": 12},
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_title(key)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")


plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=200, bbox_inches="tight")
# plt.show()
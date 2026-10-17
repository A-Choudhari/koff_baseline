"""Per-target series ranking: for each target with enough compounds, can a model trained only on that
target's other compounds rank unseen scaffolds by koff? This is the tau-RaMD-style triage use case
("which compounds in my series deserve a full SEEKR run"), and is independent of the 'which protein' signal.

Usage: python src/within_target.py [data/processed/kinetics_pairs.csv]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from direct_rate_baselines import FIG, RES, ROOT, SEED, TanimotoKNN, lgbm, load, metrics, rf

MIN_PAIRS = 20
N_FOLDS = 5


def target_cv(d, fp, desc, y, shuffle_y=False):
    rng = np.random.default_rng(SEED)
    lig = np.hstack([fp, desc])
    names = ["target_mean", "knn_tanimoto", "ridge_desc", "rf_lig", "lgbm_lig"]
    oof = {n: np.full(len(y), np.nan) for n in names}
    n_splits = min(N_FOLDS, d["scaffold"].nunique())
    if n_splits < 2:
        return None
    for tr, te in GroupKFold(n_splits, shuffle=True, random_state=SEED).split(d, groups=d["scaffold"]):
        y_tr = rng.permutation(y[tr]) if shuffle_y else y[tr]
        oof["target_mean"][te] = y_tr.mean()
        oof["knn_tanimoto"][te] = TanimotoKNN().fit(fp[tr], y_tr).predict(fp[te])
        oof["ridge_desc"][te] = make_pipeline(StandardScaler(), Ridge(1.0)).fit(desc[tr], y_tr).predict(desc[te])
        oof["rf_lig"][te] = rf().fit(lig[tr], y_tr).predict(lig[te])
        oof["lgbm_lig"][te] = lgbm().fit(lig[tr], y_tr).predict(lig[te])
    return oof


def main(path):
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df, fp, desc, _ = load(path)
    y = df["log_koff"].to_numpy()
    counts = df["target_id"].value_counts()
    rows = []
    for tid in counts[counts >= MIN_PAIRS].index:
        idx = np.where(df["target_id"] == tid)[0]
        d = df.iloc[idx].reset_index(drop=True)
        base = {"target": d["target_name"].iloc[0][:40], "n": len(idx), "n_scaffolds": d["scaffold"].nunique(),
                "y_sd": float(y[idx].std()), "source": d["source"].mode().iloc[0]}
        oof = target_cv(d, fp[idx], desc[idx], y[idx])
        if oof is None:
            continue
        for name, p in oof.items():
            rows.append({**base, "model": name, **metrics(y[idx], p)})
        scr = target_cv(d, fp[idx], desc[idx], y[idx], shuffle_y=True)
        rows.append({**base, "model": "lgbm_lig_Y_SCRAMBLED", **metrics(y[idx], scr["lgbm_lig"])})
    res = pd.DataFrame(rows)
    res.to_csv(RES / "within_target.csv", index=False)

    piv = res.pivot(index="target", columns="model", values="spearman")
    piv["n"] = res.groupby("target")["n"].first()
    piv = piv.sort_values("n", ascending=False)
    pd.set_option("display.width", 200)
    print("Within-target Spearman (scaffold-grouped CV inside each target):")
    print(piv.round(2).to_string())
    print("\nRMSE (log units) vs. target SD:")
    print(res.pivot(index="target", columns="model", values="RMSE").assign(y_sd=res.groupby("target")["y_sd"].first())
          .loc[piv.index].round(2).to_string())

    models = ["knn_tanimoto", "ridge_desc", "rf_lig", "lgbm_lig", "lgbm_lig_Y_SCRAMBLED"]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(piv) + 2))
    piv[models].plot.barh(ax=ax, width=0.8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Spearman ρ (out-of-fold, new scaffolds)", title=f"Ranking koff within a target (targets with ≥{MIN_PAIRS} pairs)")
    ax.set_yticklabels([f"{t} (n={int(n)})" for t, n in piv["n"].items()])
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "within_target_spearman.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "processed" / "kinetics_pairs.csv")

"""Stage-1 viability test: direct koff prediction from experimental BindingDB kinetics.

Question: does ligand chemistry + target identity carry learnable signal for log10(koff),
beyond (a) trivial per-target averages and (b) equilibrium affinity alone?

Usage: python src/direct_rate_baselines.py [data/processed/kinetics_pairs.csv]
Outputs go to results/.
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from features import descriptors, mol_from_smiles, morgan_fp, protein_composition, scaffold

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"
SEED = 0
N_FOLDS = 5


# ---------------------------------------------------------------- data

KOFF_RANGE = (-8, 3)  # log10 s^-1. Rows outside fail the koff/kon = Kd check by ~12 log units: curation errors.


def load(path):
    df = pd.read_csv(path)
    n0 = len(df)
    df = df[df["log_koff"].between(*KOFF_RANGE)].copy()
    print(f"dropped {n0 - len(df)} pairs with log10 koff outside {KOFF_RANGE} (curation/unit errors)")
    mols = df["smiles"].map(mol_from_smiles)
    df = df[mols.notna()].copy()
    mols = mols[mols.notna()]
    df["scaffold"] = [scaffold(m) for m in mols]
    fp = np.stack([morgan_fp(m) for m in mols])
    desc = np.stack([descriptors(m) for m in mols])
    prot = np.stack([protein_composition(s) for s in df["sequence"]])
    return df.reset_index(drop=True), fp, desc, prot


# ---------------------------------------------------------------- models

class TargetMean:
    """Predict the training mean log_koff of the same target; fall back to the global mean."""
    def fit(self, X, y, targets):
        self.global_ = y.mean()
        self.table_ = pd.Series(y).groupby(np.asarray(targets)).mean()
        return self

    def predict(self, X, targets):
        return pd.Series(np.asarray(targets)).map(self.table_).fillna(self.global_).to_numpy()


class TanimotoKNN:
    """k-nearest-neighbour regression on Morgan-fingerprint Tanimoto similarity."""
    def __init__(self, k=3):
        self.k = k

    def fit(self, fp, y):
        self.fp_, self.y_ = fp.astype(bool), y
        return self

    def predict(self, fp):
        fp = fp.astype(bool)
        inter = fp.astype(np.float32) @ self.fp_.T.astype(np.float32)
        union = fp.sum(1)[:, None] + self.fp_.sum(1)[None, :] - inter
        sim = inter / np.maximum(union, 1)
        idx = np.argsort(-sim, axis=1)[:, : self.k]
        w = np.take_along_axis(sim, idx, 1) + 1e-6
        return (self.y_[idx] * w).sum(1) / w.sum(1)


def lgbm():
    return LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
                         colsample_bytree=0.3, subsample=0.8, subsample_freq=1, random_state=SEED, verbose=-1)


def rf():
    return RandomForestRegressor(n_estimators=500, max_features=0.3, min_samples_leaf=2, n_jobs=-1, random_state=SEED)


# ---------------------------------------------------------------- evaluation

def metrics(y, p):
    return {
        "R2": r2_score(y, p),
        "RMSE": mean_squared_error(y, p) ** 0.5,
        "pearson": pearsonr(y, p)[0] if np.std(p) > 0 else np.nan,
        "spearman": spearmanr(y, p)[0] if np.std(p) > 0 else np.nan,
        "within_1log": float(np.mean(np.abs(y - p) <= 1.0)),
    }


def within_target_spearman(df, y, p, fold, min_n=6):
    """Mean Spearman within (target, fold) groups with >= min_n test compounds: the 'rank my series' use case.
    Computed per fold so a per-target constant prediction correctly scores as undefined, not as noise."""
    rhos = []
    for _, idx in df.groupby([df["target_id"], fold]).indices.items():
        if len(idx) >= min_n and np.std(y[idx]) > 0 and np.std(p[idx]) > 1e-9:
            rhos.append(spearmanr(y[idx], p[idx])[0])
    return (np.mean(rhos) if rhos else np.nan), len(rhos)


def splitter(kind, df):
    if kind == "random":
        return KFold(N_FOLDS, shuffle=True, random_state=SEED).split(df)
    groups = df["scaffold"] if kind == "scaffold" else df["target_id"]
    n = min(N_FOLDS, groups.nunique())
    return GroupKFold(n, shuffle=True, random_state=SEED).split(df, groups=groups)


def cross_validate(df, X_sets, y, split_kind, models, shuffle_y=False):
    """Out-of-fold predictions for every model; returns ({model_name: oof_pred}, fold_id_per_row)."""
    rng = np.random.default_rng(SEED)
    oof = {name: np.full(len(y), np.nan) for name in models}
    fold = np.full(len(y), -1)
    for f, (tr, te) in enumerate(splitter(split_kind, df)):
        fold[te] = f
        y_tr = rng.permutation(y[tr]) if shuffle_y else y[tr]
        for name, (feat, make) in models.items():
            if name == "target_mean":
                m = TargetMean().fit(None, y_tr, df["target_id"].iloc[tr])
                oof[name][te] = m.predict(None, df["target_id"].iloc[te])
            elif name == "global_mean":
                oof[name][te] = y_tr.mean()
            elif name.startswith("knn"):
                oof[name][te] = TanimotoKNN().fit(X_sets["fp"][tr], y_tr).predict(X_sets["fp"][te])
            else:
                X = X_sets[feat]
                oof[name][te] = make().fit(X[tr], y_tr).predict(X[te])
    return oof, fold


# ---------------------------------------------------------------- analyses

def noise_ceiling(rows_path, y_var, gross_err=3.0):
    """Pooled replicate SD from raw measurements -> best achievable R2 given assay noise.
    Replicate pairs whose values span > gross_err log units are curation errors, not assay noise;
    they are counted and excluded. Median replicate range is ~0.02, i.e. many 'replicates' are the same
    measurement re-deposited, so the cross-source figure is the honest one."""
    rows = pd.read_csv(rows_path, usecols=["inchikey", "target_id", "log_koff", "koff_qual", "doi"])
    rows = rows[(rows["koff_qual"].fillna("") == "") & rows["log_koff"].between(*KOFF_RANGE)]
    stats = rows.groupby(["inchikey", "target_id"]).agg(
        n=("log_koff", "size"), var=("log_koff", "var"), n_src=("doi", "nunique"),
        rng=("log_koff", lambda s: s.max() - s.min()))
    rep = stats[stats["n"] > 1]
    gross = rep[rep["rng"] > gross_err]
    rep = rep[rep["rng"] <= gross_err]
    out = {"n_replicate_pairs": int(len(rep) + len(gross)), "n_gross_discordant_excluded": int(len(gross)),
           "median_replicate_range": float(rep["rng"].median())}
    for label, r in [("all_replicates", rep), ("cross_source", rep[rep["n_src"] > 1])]:
        if len(r) < 3:
            continue
        pooled_var = (r["var"] * (r["n"] - 1)).sum() / (r["n"] - 1).sum()
        out[label] = {"n_pairs": int(len(r)), "pooled_sd": float(np.sqrt(pooled_var)),
                      "max_R2": float(1 - pooled_var / y_var)}
    return out


def affinity_analysis(df, X_sets, y):
    """On pairs with an affinity value (Kd, else Ki, else IC50): how much of log_koff is just affinity,
    and does chemistry add anything beyond it?"""
    sub = df["p_aff"].notna().to_numpy()
    if sub.sum() < 50:
        return None, {"n_with_affinity": int(sub.sum())}
    d = df[sub].reset_index(drop=True)
    ys = y[sub]
    # Where Kd is reported alongside kon/koff, is it merely koff/kon recomputed? (circular feature check)
    kd = d["pKd"].notna() & d["log_kon"].notna()
    derived = (np.abs(d["pKd"] + d["log_koff"] - d["log_kon"]) < 0.05)[kd]
    paff = d[["p_aff"]].to_numpy()
    feats = {"fp": X_sets["fp"][sub], "lig_prot": X_sets["lig_prot"][sub],
             "lig_prot_aff": np.hstack([X_sets["lig_prot"][sub], paff]), "aff": paff}
    models = {
        "global_mean": (None, None),
        "target_mean": (None, None),
        "affinity_only_linear": ("aff", LinearRegression),
        "lgbm_lig_prot": ("lig_prot", lgbm),
        "lgbm_lig_prot+affinity": ("lig_prot_aff", lgbm),
    }
    rows = []
    for split in ["random", "scaffold", "target"]:
        oof, fold = cross_validate(d, feats, ys, split, models)
        for name, p in oof.items():
            wt, n_t = within_target_spearman(d, ys, p, fold)
            rows.append({"subset": "has_affinity", "split": split, "model": name, "n": len(ys),
                         **metrics(ys, p), "within_target_spearman": wt, "n_targets_ranked": n_t})
    info = {"n_with_affinity": int(sub.sum()), "affinity_types": d["aff_type"].value_counts().to_dict(),
            "n_Kd_with_kon": int(kd.sum()),
            "frac_Kd_derived_from_kinetics": float(derived.mean()) if kd.any() else None,
            "pearson_affinity_vs_logkoff": float(pearsonr(d["p_aff"], ys)[0])}
    return pd.DataFrame(rows), info


def learning_curve(df, X, y, split_kind="scaffold", fracs=(0.1, 0.2, 0.35, 0.5, 0.75, 1.0)):
    rng = np.random.default_rng(SEED)
    out = []
    for frac in fracs:
        oof = np.full(len(y), np.nan)
        for tr, te in splitter(split_kind, df):
            sub = rng.choice(tr, max(20, int(frac * len(tr))), replace=False)
            oof[te] = lgbm().fit(X[sub], y[sub]).predict(X[te])
        out.append({"train_frac": frac, "n_train": int(frac * len(y) * (N_FOLDS - 1) / N_FOLDS),
                    "split": split_kind, **metrics(y, oof)})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- plots

def plot_dataset(df):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(df["log_koff"], bins=40, color="#4C72B0")
    ax[0].set(xlabel="log10 koff (s$^{-1}$)", ylabel="ligand-target pairs", title="Distribution of experimental koff")
    sec = ax[0].secondary_xaxis("top", functions=(lambda x: -x, lambda x: -x))
    sec.set_xlabel("log10 residence time τ (s)")
    top = df["target_name"].str.slice(0, 35).value_counts().head(15)[::-1]
    ax[1].barh(top.index, top.values, color="#55A868")
    ax[1].set(xlabel="pairs", title="Data concentration: top 15 targets")
    ax[1].tick_params(axis="y", labelsize=7)
    k = df.dropna(subset=["log_kon"])
    ax[2].scatter(k["log_kon"], k["log_koff"], s=6, alpha=0.4, color="#C44E52")
    ax[2].set(xlabel="log10 kon (M$^{-1}$s$^{-1}$)", ylabel="log10 koff (s$^{-1}$)", title="kon vs koff")
    fig.tight_layout()
    fig.savefig(FIG / "dataset_overview.png", dpi=150)
    plt.close(fig)


def plot_parity(y, oof_by_split, model):
    fig, ax = plt.subplots(1, len(oof_by_split), figsize=(5 * len(oof_by_split), 4.5))
    for a, (split, oof) in zip(np.atleast_1d(ax), oof_by_split.items()):
        p = oof[model]
        a.scatter(y, p, s=6, alpha=0.4)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1], color="grey", alpha=0.12)
        m = metrics(y, p)
        a.set(xlabel="measured log10 koff", ylabel="predicted", title=f"{model} | {split} split\nR²={m['R2']:.2f}  ρ={m['spearman']:.2f}  RMSE={m['RMSE']:.2f}")
    fig.tight_layout()
    fig.savefig(FIG / f"parity_{model}.png", dpi=150)
    plt.close(fig)


def plot_summary(res, ceiling):
    piv = res[res["subset"] == "all"].pivot(index="model", columns="split", values="R2")[["random", "scaffold", "target"]]
    piv = piv.sort_values("random")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    piv.plot.barh(ax=ax, color=["#4C72B0", "#DD8452", "#C44E52"])
    ceilings = [v["max_R2"] for v in ceiling.values() if isinstance(v, dict)]
    if ceilings:
        ax.axvline(min(ceilings), color="k", ls=":", label="noise ceiling (approx)")
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="out-of-fold R²", title="Direct koff prediction: R² by model and split difficulty")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "r2_by_split.png", dpi=150)
    plt.close(fig)


def plot_learning_curve(lc):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for split, g in lc.groupby("split"):
        ax[0].plot(g["n_train"], g["RMSE"], "o-", label=split)
        ax[1].plot(g["n_train"], g["spearman"], "o-", label=split)
    for a, lab in zip(ax, ["RMSE (log units)", "Spearman ρ"]):
        a.set(xscale="log", xlabel="training pairs", ylabel=lab)
        a.legend()
    fig.suptitle("Learning curve (LightGBM, ligand+protein features)")
    fig.tight_layout()
    fig.savefig(FIG / "learning_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(path):
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    df, fp, desc, prot = load(path)
    y = df["log_koff"].to_numpy()
    print(f"{len(df)} pairs | {df['inchikey'].nunique()} ligands | {df['target_id'].nunique()} targets | "
          f"{df['scaffold'].nunique()} scaffolds | y mean {y.mean():.2f} sd {y.std():.2f}")
    X_sets = {"fp": fp, "desc": desc, "lig": np.hstack([fp, desc]), "lig_prot": np.hstack([fp, desc, prot])}

    models = {
        "global_mean": (None, None),
        "target_mean": (None, None),
        "knn_tanimoto": ("fp", None),
        "ridge_desc": ("desc", lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        "rf_lig": ("lig", rf),
        "rf_lig_prot": ("lig_prot", rf),
        "lgbm_lig": ("lig", lgbm),
        "lgbm_lig_prot": ("lig_prot", lgbm),
    }
    rows, oof_by_split = [], {}
    for split in ["random", "scaffold", "target"]:
        print(f"  CV split = {split}")
        oof, fold = cross_validate(df, X_sets, y, split, models)
        oof_by_split[split] = oof
        for name, p in oof.items():
            wt, n_t = within_target_spearman(df, y, p, fold)
            rows.append({"subset": "all", "split": split, "model": name, "n": len(y), **metrics(y, p),
                         "within_target_spearman": wt, "n_targets_ranked": n_t})
    scr, _ = cross_validate(df, X_sets, y, "random", {"lgbm_lig_prot": ("lig_prot", lgbm)}, shuffle_y=True)
    rows.append({"subset": "all", "split": "random", "model": "lgbm_lig_prot_Y_SCRAMBLED", "n": len(y),
                 **metrics(y, scr["lgbm_lig_prot"])})
    res = pd.DataFrame(rows)

    aff_res, aff_info = affinity_analysis(df, X_sets, y)
    if aff_res is not None:
        res = pd.concat([res, aff_res], ignore_index=True)
    res.to_csv(RES / "metrics.csv", index=False)

    ceiling = noise_ceiling(Path(path).with_name("kinetics_rows.csv"), y.var())
    lc = pd.concat([learning_curve(df, X_sets["lig_prot"], y, s) for s in ["random", "scaffold"]])
    lc.to_csv(RES / "learning_curve.csv", index=False)

    plot_dataset(df)
    plot_parity(y, oof_by_split, "lgbm_lig_prot")
    plot_summary(res, ceiling)
    plot_learning_curve(lc)

    pd.set_option("display.width", 200)
    cols = ["subset", "split", "model", "R2", "RMSE", "spearman", "within_1log", "within_target_spearman"]
    print(res[[c for c in cols if c in res]].round(3).to_string(index=False))
    print("\nNoise ceiling:", ceiling)
    print("Affinity analysis:", aff_info)
    print("\nLearning curve:\n", lc.round(3).to_string(index=False))
    pd.Series({"noise_ceiling": ceiling, "affinity": aff_info}).to_json(RES / "extra_info.json", indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "processed" / "kinetics_pairs.csv")

# Direct koff prediction from BindingDB — Stage-1 viability test

Context: the "Binding Kinetics at Scale" / KCFM proposal (Amaro, Yu, Gilson; UCSD–NVIDIA summit, Sept 2026)
says the first, simplest baseline is **direct rate prediction**: treat off-rates as scalar labels and see
how far structure + chemistry alone can go. This repo runs that baseline on the *experimental* kinetics
already in BindingDB, before any SEEKR data exist.

## Question
Does ligand chemistry + target identity carry learnable signal for `log10(koff)` in BindingDB's
experimental kinetic subset, and does that signal survive the splits that matter (new scaffold, new target)?

## Pipeline
| Step | Script | Output |
|---|---|---|
| 1. Download `BindingDB_All_<release>_tsv.zip` | manual / curl | `data/raw/` |
| 2. Stream the TSV, keep rows with kon/koff, aggregate replicates per ligand–target pair | `src/extract_kinetics.py` | `data/processed/kinetics_rows.csv`, `kinetics_pairs.csv` |
| 3. Featurize (Morgan FP 2048 + 11 RDKit descriptors + protein AA composition), run baselines under 3 CV schemes | `src/direct_rate_baselines.py` | `results/metrics.csv`, `results/learning_curve.csv`, `results/figures/*.png` |

```powershell
.\.venv\Scripts\python.exe src\extract_kinetics.py data\raw\BindingDB_All_202610_tsv.zip
.\.venv\Scripts\python.exe src\direct_rate_baselines.py
```

## Models (weakest → strongest)
- `global_mean` – predict the dataset mean. Floor.
- `target_mean` – predict the per-target mean. Tests whether "signal" is just *which protein*.
- `knn_tanimoto` – 3-NN on fingerprint similarity. Tests whether signal is just analog lookup.
- `ridge_desc` – linear model on 11 physchem descriptors.
- `rf_lig`, `lgbm_lig` – tree ensembles on ligand features only.
- `rf_lig_prot`, `lgbm_lig_prot` – tree ensembles on ligand + protein features.
- `lgbm_lig_prot_Y_SCRAMBLED` – same model, labels permuted. Must land near R² ≈ 0 or the CV leaks.

## Splits (easiest → hardest)
- `random` – 5-fold. Upper bound; inflated by near-duplicate analogs.
- `scaffold` – GroupKFold on Murcko scaffold. "New chemotype on a known target."
- `target` – GroupKFold on UniProt ID. "Never-seen protein." Expected to be hard.

## Metrics
R², RMSE (log10 units), Pearson, Spearman, fraction within 1 log unit, and
`within_target_spearman` (mean rank correlation inside targets with ≥8 test compounds — the
"rank my series" use case that matters for triage).

## Reference points
- **Noise ceiling**: pooled SD of replicate measurements of the same pair → max achievable R².
- **Affinity-only**: on pairs with measured Kd, does pKd alone predict koff? Does adding chemistry beat it?
  Also reports what fraction of Kd values are just koff/kon recomputed (circular feature check).
- **Learning curve**: RMSE vs training-set size, to extrapolate what thousands of SEEKR labels could buy.

## Reading the result
| Outcome | Interpretation |
|---|---|
| scaffold-split Spearman ≥ ~0.5 and clearly above `target_mean` | chemistry carries kinetic signal → direct prediction viable, worth scaling |
| random good, scaffold ≈ `target_mean` | model memorises analogs; signal is mostly "which target" → need more diverse labels (the SEEKR corpus) |
| everything ≈ scrambled baseline | no learnable signal at this data size → scalar-label approach not viable; argues for KCFM-style structured labels |
| learning curve still falling at full size | data-limited → quantifies the value of generating more labels |

# Results — direct koff prediction on BindingDB experimental kinetics (release 2026-10)

Run date: 2026-10-16. Full tables in `results/metrics.csv`, `results/within_target.csv`,
`results/learning_curve.csv`; figures in `results/figures/`.

## Dataset
- 3,243,660 BindingDB rows scanned → 1,019 rows with koff (8 censored), 1,003 with kon, 246 with both.
- 788 unique ligand–target pairs with an exact koff; 681 ligands, 87 targets, 103 pairs with a PDB complex.
- **Extreme concentration**: FAP (183 pairs, all from one patent) + neutrophil elastase (173, 7 papers) = 46 %.
  18 targets have ≥ 8 pairs and cover 631/788 pairs; the other 69 targets have < 8 each.
- log10 koff: mean −2.7, SD 1.5 (residence times from seconds to days).

## Curation errors found (and removed)
- 12 Carbonic anhydrase 2 rows: koff 10⁴–10⁶ s⁻¹ with kon ≈ 0.2 M⁻¹s⁻¹; koff/kon disagrees with the reported Kd
  by **11.75 log units** → kon/koff units swapped at curation.
- 3 neutrophil elastase rows at koff 10⁶–10⁷ s⁻¹ whose duplicates are at 10⁻⁶ → exponent sign error.
- Filter applied: 10⁻⁸ ≤ koff ≤ 10³ s⁻¹; 15 pairs dropped (772 remain, 85 targets, 291 scaffolds).
- Of 17 pairs reporting Kd together with kon and koff, 88 % have Kd = koff/kon exactly — Kd is *derived*, so it
  must not be used as an independent feature.

## Noise ceiling
177 replicated pairs; median replicate range **0.01 log** (most "replicates" are the same number re-deposited).
Excluding the 3 gross-discordant pairs: pooled replicate SD 0.33 log (174 pairs), cross-source SD 0.27 log
(76 pairs) → max achievable R² ≈ 0.95. The models are **not** at the noise limit.

## Main table (out-of-fold, 772 pairs, log10 koff)

| split | model | R² | RMSE | Spearman | within 1 log | within-target ρ |
|---|---|---|---|---|---|---|
| random | global_mean | 0.00 | 1.53 | −0.07 | 0.38 | — |
| random | **target_mean** | 0.59 | 0.97 | 0.73 | 0.76 | — |
| random | knn_tanimoto | 0.60 | 0.96 | 0.79 | 0.82 | 0.43 |
| random | lgbm_lig_prot | **0.70** | 0.84 | 0.84 | 0.83 | 0.57 |
| random | lgbm_lig_prot, labels scrambled | −0.27 | 1.72 | 0.07 | 0.41 | — |
| scaffold | target_mean | 0.51 | 1.07 | 0.72 | 0.74 | — |
| scaffold | rf_lig_prot | **0.57** | 1.00 | 0.78 | 0.76 | 0.49 |
| scaffold | lgbm_lig_prot | 0.58 | 0.99 | 0.77 | 0.77 | 0.34 |
| target | target_mean (= global) | −0.11 | 1.60 | −0.42 | 0.34 | — |
| target | best (rf_lig_prot) | −0.24 | 1.69 | 0.10 | 0.37 | 0.31 |

## Within-target series ranking (scaffold-grouped CV inside one target, Spearman ρ)

| target | n | best model | ρ | scrambled ρ | RMSE best / SD |
|---|---|---|---|---|---|
| FAP (patent) | 183 | ridge_desc | 0.33 | −0.11 | 0.58 / 0.61 |
| Neutrophil elastase | 173 | knn_tanimoto | 0.51 | −0.06 | 0.92 / 1.11 |
| HIV-1 protease (Gag-Pol) | 39 | ridge_desc | 0.75 | −0.17 | 1.16 / 1.44 |
| CDK2 | 26 | rf_lig | 0.69 | 0.11 | 0.55 / 0.77 |
| Carbonic anhydrase 2 | 23 | knn_tanimoto | 0.53 | 0.20 | 0.50 / 0.58 |
| S1P receptor 2 | 21 | ridge_desc | 0.32 | −0.06 | 0.26 / 0.29 |
| Carbonic anhydrase 7 | 20 | ridge_desc | 0.52 | −0.44 | 1.23 / 1.03 |

## Affinity control (181 pairs with Ki/Kd)
Affinity alone: R² 0.03 random, −0.17 scaffold; Pearson(affinity, log koff) = −0.24.
Chemistry model: R² 0.53 / 0.25; adding affinity as a feature changes R² by < 0.01.
→ koff is not recoverable from affinity; the kinetic signal is distinct.

## Learning curve (LightGBM, ligand + protein)
Scaffold-split RMSE still falling at full size: 1.43 (61 train) → 1.09 (216) → 0.97 (617). No plateau.

## Reading
1. **There is real, learnable kinetic signal** — scrambled control ≈ 0, real model R² 0.70 random / 0.58 scaffold,
   far below the 0.95 noise ceiling, so the limit is data, not label quality.
2. **Most of the pooled signal is "which protein"**: `target_mean` alone gets R² 0.59 / 0.51. Chemistry adds
   ~0.1 R² and ~0.1 RMSE on top. Pooled R² overstates what a model knows about ligand kinetics.
3. **Zero transfer to unseen targets** at this size (85 targets, two of which are half the data). Any
   "foundation kinetics model" claim needs target diversity the experimental set cannot supply.
4. **Series ranking within a known target works moderately** (ρ 0.3–0.75 on new scaffolds vs ≈ 0 scrambled),
   but absolute error is barely below the target's own SD — good enough for triage, not for prediction.
5. **Not an affinity proxy**: affinity explains ~0 % of koff variance here, confirming the proposal's premise.
6. **Data-limited**: learning curve has no plateau, which is the quantitative argument for a larger corpus —
   with the caveat that the corpus must add *targets and scaffolds*, not more analogs on FAP/elastase.

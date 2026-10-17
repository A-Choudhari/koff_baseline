"""Stream the full BindingDB TSV and keep every row that carries a kon or koff value.

Usage: python src/extract_kinetics.py data/raw/BindingDB_All_202610_tsv.zip
Writes data/processed/kinetics_rows.csv (one row per measurement)
   and data/processed/kinetics_pairs.csv (one row per ligand-target pair, replicates aggregated).
"""
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "processed"

# Logical name -> substring(s) used to find the real BindingDB header (headers drift between releases).
WANTED = {
    "reactant_set_id": ["BindingDB Reactant_set_id"],
    "smiles": ["Ligand SMILES"],
    "inchikey": ["Ligand InChI Key"],
    "monomer_id": ["BindingDB MonomerID"],
    "target_name": ["Target Name"],
    "ki_nM": ["Ki (nM)"],
    "ic50_nM": ["IC50 (nM)"],
    "kd_nM": ["Kd (nM)"],
    "kon": ["kon ("],
    "koff": ["koff ("],
    "pH": ["pH"],
    "temp_C": ["Temp (C)"],
    "doi": ["Article DOI"],
    "patent": ["Patent Number"],
    "pdb_complex": ["PDB ID(s) for Ligand-Target Complex"],
    "n_chains": ["Number of Protein Chains in Target"],
    "sequence": ["BindingDB Target Chain Sequence"],
    "uniprot": ["UniProt (SwissProt) Primary ID of Target Chain"],
    "uniprot_trembl": ["UniProt (TrEMBL) Primary ID of Target Chain"],
}


def resolve_columns(header):
    mapping = {}
    for key, needles in WANTED.items():
        for needle in needles:
            hits = [h for h in header if h == needle] or [h for h in header if h.startswith(needle)]
            if hits:
                mapping[key] = hits[0]  # first match = chain 1 for multi-chain columns
                break
    return mapping


def parse_value(series):
    """BindingDB values look like '12', '>10000', '<0.1', ' 3.5E-4'. Return (float, qualifier)."""
    s = series.astype(str).str.strip()
    qual = s.str.extract(r"^([<>]=?|~)?", expand=False).fillna("")
    num = pd.to_numeric(s.str.replace(r"^[<>=~]+", "", regex=True).str.strip(), errors="coerce")
    num = num.where(num > 0)
    return num, qual


def main(zip_path):
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith(".tsv"))
        with zf.open(name) as fh:
            header = fh.readline().decode("utf-8", "replace").rstrip("\r\n").split("\t")
        cols = resolve_columns(header)
        missing = set(WANTED) - set(cols)
        print(f"{len(header)} columns; resolved {len(cols)}; missing: {sorted(missing) or 'none'}")
        if "koff" not in cols or "kon" not in cols:
            print("kon/koff headers not found. Header sample:", header[:60])
            sys.exit(1)

        keep, n_total = [], 0
        with zf.open(name) as fh:
            reader = pd.read_csv(
                fh, sep="\t", usecols=list(cols.values()), dtype=str, chunksize=200_000,
                on_bad_lines="skip", quoting=3, encoding="utf-8", encoding_errors="replace",
            )
            for chunk in reader:
                n_total += len(chunk)
                mask = chunk[cols["koff"]].notna() | chunk[cols["kon"]].notna()
                if mask.any():
                    keep.append(chunk[mask])
                print(f"  scanned {n_total:,} rows, kinetic rows so far {sum(map(len, keep)):,}", end="\r")
    print()

    df = pd.concat(keep).rename(columns={v: k for k, v in cols.items()})
    for c in ["koff", "kon", "kd_nM", "ki_nM", "ic50_nM"]:
        if c in df:
            df[c], df[c + "_qual"] = parse_value(df[c])
    for c in ["pH", "temp_C"]:
        if c in df:
            df[c] = pd.to_numeric(df[c].str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

    df["log_koff"] = np.log10(df["koff"])
    df["log_kon"] = np.log10(df["kon"])
    df["pKd"] = 9 - np.log10(df["kd_nM"])
    df["pKi"] = 9 - np.log10(df["ki_nM"])
    df["pIC50"] = 9 - np.log10(df["ic50_nM"])
    df["target_id"] = df.get("uniprot").fillna(df.get("uniprot_trembl")).fillna(df["target_name"])
    df["has_pdb"] = df["pdb_complex"].notna() & (df["pdb_complex"].str.strip() != "")
    df["source"] = np.where(df["doi"].notna(), "article", np.where(df["patent"].notna(), "patent", "other"))
    df.to_csv(OUT / "kinetics_rows.csv", index=False)
    pairs = build_pairs(df)
    print(f"Total BindingDB rows scanned: {n_total:,}")
    report(df, pairs)


def build_pairs(df):
    """Aggregate replicates per ligand-target pair. Keep the spread: it is the experimental noise floor."""
    df = df.copy()
    df["koff_qual"] = df["koff_qual"].fillna("")
    for p_col, nm_col in [("pKd", "kd_nM"), ("pKi", "ki_nM"), ("pIC50", "ic50_nM")]:
        if p_col not in df:
            df[p_col] = 9 - np.log10(df[nm_col])
    # Affinity proxy for the 'is it just affinity?' control: prefer measured Kd, then Ki, then IC50.
    df["p_aff"] = df["pKd"].fillna(df["pKi"]).fillna(df["pIC50"])
    df["aff_type"] = np.select([df["pKd"].notna(), df["pKi"].notna(), df["pIC50"].notna()],
                               ["Kd", "Ki", "IC50"], default=None)
    exact = df[(df["koff_qual"] == "") & df["log_koff"].notna()]
    pairs = exact.groupby(["inchikey", "target_id"]).agg(
        smiles=("smiles", "first"),
        target_name=("target_name", "first"),
        sequence=("sequence", "first"),
        log_koff=("log_koff", "median"),
        log_koff_sd=("log_koff", "std"),
        n_meas=("log_koff", "size"),
        log_kon=("log_kon", "median"),
        pKd=("pKd", "median"),
        p_aff=("p_aff", "median"),
        aff_type=("aff_type", "first"),
        temp_C=("temp_C", "median"),
        has_pdb=("has_pdb", "max"),
        n_sources=("doi", "nunique"),
        source=("source", "first"),
    ).reset_index()
    pairs.to_csv(OUT / "kinetics_pairs.csv", index=False)
    return pairs


def report(df, pairs):
    print(f"Rows with koff: {df['koff'].notna().sum():,} (censored: {(df['koff_qual'] != '').sum():,})")
    print(f"Rows with kon:  {df['kon'].notna().sum():,}")
    print(f"Rows with both kon & koff: {(df['koff'].notna() & df['kon'].notna()).sum():,}")
    print(f"Rows with koff & a PDB complex: {(df['koff'].notna() & df['has_pdb']).sum():,}")
    print(f"Unique ligand-target pairs with exact koff: {len(pairs):,}")
    print(f"  unique ligands {pairs['inchikey'].nunique():,}, unique targets {pairs['target_id'].nunique():,}")
    print(f"  pairs with a PDB complex: {pairs['has_pdb'].sum():,}")
    print(f"  pairs with replicate measurements: {(pairs['n_meas'] > 1).sum():,}")


if __name__ == "__main__":
    if "--from-rows" in sys.argv:  # rebuild the pair table from kinetics_rows.csv without rescanning 9 GB
        rows = pd.read_csv(OUT / "kinetics_rows.csv", dtype={"koff_qual": str})
        report(rows, build_pairs(rows))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/BindingDB_All_202610_tsv.zip")

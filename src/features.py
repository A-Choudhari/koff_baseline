"""Ligand and protein featurization for direct kinetic-rate prediction."""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

DESCRIPTOR_FNS = {
    "MolWt": Descriptors.MolWt,
    "LogP": Descriptors.MolLogP,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "HBD": rdMolDescriptors.CalcNumHBD,
    "HBA": rdMolDescriptors.CalcNumHBA,
    "RotB": rdMolDescriptors.CalcNumRotatableBonds,
    "Rings": rdMolDescriptors.CalcNumRings,
    "AromRings": rdMolDescriptors.CalcNumAromaticRings,
    "FracSP3": rdMolDescriptors.CalcFractionCSP3,
    "HeavyAtoms": lambda m: m.GetNumHeavyAtoms(),
    "Charge": lambda m: Chem.GetFormalCharge(m),
}


def mol_from_smiles(smi):
    if not isinstance(smi, str):
        return None
    return Chem.MolFromSmiles(smi)


def morgan_fp(mol, radius=2, n_bits=2048):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(mol).astype(np.uint8)


def descriptors(mol):
    return np.array([fn(mol) for fn in DESCRIPTOR_FNS.values()], dtype=float)


def scaffold(mol):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "acyclic"
    except Exception:
        return "acyclic"


def protein_composition(seq):
    """Amino-acid composition (20) + coarse length: a deliberately simple target descriptor."""
    if not isinstance(seq, str) or not seq:
        return np.zeros(len(AMINO_ACIDS) + 1)
    seq = seq.upper()
    counts = np.array([seq.count(a) for a in AMINO_ACIDS], dtype=float)
    return np.append(counts / max(len(seq), 1), np.log10(len(seq)))

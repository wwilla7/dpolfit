#!/usr/bin/env python

"""
Train and fit AM1-BCC-dPol charges with dPol polarizabilities.
"""
import os
import shutil
import json
import uuid
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from glob import glob
import numpy as np
from typing import List, Dict
from scipy.spatial.distance import cdist
from dataclasses import dataclass
from dpolfit.fitting.respdpol import coulomb_scaling, Mol
import logging
import sys
import ray

logger = logging.getLogger(__name__)

bohr_2_angstrom = 0.529177
DEBUG = 1


@dataclass
class BCCParameter:
    smarts: str
    value: float
    pid: int


@dataclass
class BCCLocator:
    pattern: str
    indices: tuple


@dataclass
class BondChargeCorrection:
    assignment: np.array
    correction: np.array
    applied_correction: np.array


def compute_am1(rdmol):
    n_atoms = rdmol.GetNumAtoms()
    formal_charge = Chem.GetFormalCharge(rdmol)
    cwd = os.getcwd()
    tmp_path = os.path.join(cwd, str(uuid.uuid4()))
    os.makedirs(tmp_path)
    os.chdir(tmp_path)
    Chem.MolToPDBFile(rdmol, "molecule.pdb")
    subprocess.run(
        [
            "antechamber",
            "-i",
            "molecule.pdb",
            "-fi",
            "pdb",
            "-o",
            "molecule.mol2",
            "-fo",
            "mol2",
            "-c",
            "mul",
            "-nc",
            str(formal_charge),
        ]
    )
    subprocess.run(
        [
            "antechamber",
            "-i",
            "molecule.mol2",
            "-fi",
            "mol2",
            "-o",
            "molecule.mol2",
            "-fo",
            "mol2",
            "-c",
            "wc",
            "-nc",
            str(formal_charge),
            "-cf",
            "charges.dat",
        ]
    )
    with open("charges.dat", "r") as f:
        charges = f.read().split()
    charges = np.array(charges).astype(float)
    charges += (formal_charge - charges.sum()) / n_atoms
    os.chdir(cwd)
    if DEBUG == 1:
        shutil.rmtree(tmp_path)
    return charges


def find_bccs(mol: Chem.Mol, bccs: list):
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SetAromaticity(mol, model=Chem.AromaticityModel.AROMATICITY_MDL)
    ret = []
    for bcc in bccs:
        pt = Chem.MolFromSmarts(bcc)
        if mol.HasSubstructMatch(pt):
            ret.append(BCCLocator(pattern=bcc, indices=mol.GetSubstructMatches(pt)))

    return ret


def assignment_matrix(mol: Chem.Mol, BCCs: List[BCCParameter]):
    n_atoms = mol.GetNumAtoms()
    n_bccs = len(BCCs)
    location = {p.smarts: idx for idx, p in enumerate(BCCs)}
    bcc_list = location.keys()
    corrections = np.array([p.value for p in BCCs])

    assignment = np.zeros((n_atoms, n_bccs))
    counts = np.zeros((n_atoms, n_bccs))

    matches = find_bccs(mol, bcc_list)
    matched = []
    for match in matches:
        pattern = match.pattern
        indices = match.indices
        for index in indices:
            reverse = index[::-1]
            if index in matched or reverse in matched:
                continue
            this_match_idx = location[pattern]
            assignment[index[0], this_match_idx] += 1
            assignment[index[1], this_match_idx] -= 1

            counts[index[0], this_match_idx] += 1
            counts[index[1], this_match_idx] += 1

            matched.extend([index, reverse])

    # check not assigned
    not_assigned = np.where(~counts.any(axis=1))[0]
    if len(not_assigned) > 0:
        raise ValueError(f"BCC not fully assigned for {not_assigned}")

    # each BCCs sums to zero
    wrong_assignment = assignment.sum(axis=0).nonzero()[0]
    if len(wrong_assignment) > 0:
        raise ValueError(
            f"wrong assignment in {[location[i] for i in wrong_assignment]}"
        )

    ret = BondChargeCorrection(
        assignment=assignment,
        correction=corrections,
        applied_correction=assignment @ corrections,
    )

    return ret


def fit(mol: Mol, BCCs: List[BCCParameter]):

    rdmol = Chem.MolFromSmiles(mol.mapped_smiles)
    rdmol = Chem.AddHs(rdmol)
    n_atoms = rdmol.GetNumAtoms()
    AllChem.Compute2DCoords(rdmol)
    conf = rdmol.GetConformer()
    for i in range(rdmol.GetNumAtoms()):
        xyz = mol.coordinates[i] * bohr_2_angstrom
        conf.SetAtomPosition(i, Point3D(*xyz))

    precharges = compute_am1(rdmol).reshape(-1)
    # offmol = mol.offmol
    # offmol.assign_partial_charges("am1-mulliken")
    # precharges = offmol.partial_charges.m_as("elementary_charge")

    n_bccs = len(BCCs)
    location = {p.smarts: idx for idx, p in enumerate(BCCs)}
    bcc_list = location.keys()

    crds = mol.coordinates
    grid = mol.grid

    r_ij = -(grid - crds[:, None])
    r_ij0 = cdist(crds, grid, metric="euclidean")
    r_ij1 = np.power(r_ij0, -1)
    r_ij3 = np.power(r_ij0, -3)

    r_jk = crds - crds[:, None]
    r_jk1 = cdist(crds, crds, metric="euclidean")
    r_jk3 = np.power(r_jk1, -3, where=r_jk1 != 0)

    coulomb14scale_matrix = coulomb_scaling(rdmol, coulomb14scale=0.5)

    drjk = np.zeros((n_atoms, n_atoms, 3))
    for k in range(n_atoms):
        drjk[k] = r_jk[k] * (r_jk3[k] * coulomb14scale_matrix[k]).reshape(-1, 1)

    A = np.zeros((n_bccs, n_bccs))
    B = np.zeros(n_bccs)

    matched_bccs = find_bccs(rdmol, bcc_list)
    param_dict = {
        r.pattern: list(set([atom for pair in r.indices for atom in pair]))
        for r in matched_bccs
    }

    esps = mol.data_esp.reshape(-1)
    precharge_esp = np.dot(r_ij1.T, precharges)
    efield = np.zeros((n_atoms, 3))
    for k in range(n_atoms):
        efield[k] = np.dot(precharges, drjk[k])

    deij = np.einsum("jm, jim->ji", efield, r_ij) * r_ij3
    desp = np.dot(mol.polarizabilities, deij)

    vdiff = esps - precharge_esp - desp

    ret_assignment = assignment_matrix(rdmol, BCCs)
    assignment = ret_assignment.assignment

    D_ijE = np.zeros((n_atoms, n_bccs, grid.shape[0]))
    for i in range(n_atoms):
        for j in range(n_bccs):
            D_ijE[i, j] = assignment[i, j] * r_ij1[i]

    for k, (smirks_k, bccs_k) in enumerate(param_dict.items()):
        logger.info(f"Processing {smirks_k}...")

        Cik = D_ijE[bccs_k].sum(axis=1).sum(axis=0)
        B[location[smirks_k]] += np.dot(Cik, vdiff).item()

        for l, (smirks_l, bccs_l) in enumerate(param_dict.items()):
            if k > l:
                continue
            Cil = D_ijE[bccs_l].sum(axis=1).sum(axis=0)
            A[location[smirks_k], location[smirks_l]] += np.dot(Cil, Cik)

    A += A.T - np.diag(A.diagonal())

    return A, B


def training(traingset_path: str, BCCs: List[BCCParameter], off_ff, num_cpus: int = 8):
    cwd = os.getcwd()
    if num_cpus > 1:
        ray.init(_temp_dir="/tmp/ray_tmp", num_gpus=0, num_cpus=num_cpus)
        fit_remote = ray.remote(num_cpus=1, num_gpus=0)(fit)
        fit_method = {"fit": fit_remote.remote}
    else:
        fit_method = {"fit": fit}

    ndim = len(BCCs)
    A = np.zeros((ndim, ndim))
    B = np.zeros(ndim)
    location = {p.smarts: idx for idx, p in enumerate(BCCs)}
    molecules = glob(os.path.join(traingset_path, "mol*"))
    mols = [Mol(data_path=f, off_ff=off_ff) for f in molecules]
    ret = [fit_method["fit"](mol=mol, BCCs=BCCs) for mol in mols]
    if num_cpus > 1:
        ret = ray.get(ret)
    A += sum([r[0] for r in ret])
    B += sum([r[1] for r in ret])

    if np.where(B == 0)[0].size > 0:
        not_fitted = np.where(B == 0)[0]

        for i in sorted(not_fitted, reverse=True):
            nft = BCCs.pop(i)
            logger.info(f"Parameter {nft.smarts} not fitted")

        fitted_smirks = [p.smarts for p in BCCs]
        fitted_positions = [location[p] for p in fitted_smirks]

        A = A[tuple(np.meshgrid(fitted_positions, fitted_positions))]
        B = B[fitted_positions]

    else:
        pass

    np.save(os.path.join(cwd, "A.npy"), A)
    np.save(os.path.join(cwd, "B.npy"), B)

    logger.debug(f"A: {A}")
    logger.debug(f"B: {B}")

    ret = np.linalg.solve(A, B)

    results = []
    for p, v in zip(BCCs, ret):
        p.value = v
        results.append(p.__dict__)

    json.dump(results, open(os.path.join(cwd, "results.json"), "w"), indent=2)
    return BCCs


if __name__ == "__main__":
    from openff.toolkit import ForceField

    original_bccs = json.load(open("original-am1-bcc.json", "r"))
    logging.basicConfig(
        filename="training.log",
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    ff = ForceField("custom.offxml", allow_cosmetic_attributes=True, load_plugins=True)
    BCCs = [
        BCCParameter(smarts=r["smirks"], value=0.0, pid=idx)
        for idx, r in enumerate(original_bccs)
    ]
    training_data = "/home/wwilla/data_scratch/data_mikoyan/data_water/data_solv/training/trainingset/data"

    results = training(traingset_path=training_data, BCCs=BCCs, off_ff=ff)

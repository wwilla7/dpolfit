#!/usr/bin/env python

"""
Train and fit AM1-BCC-dPol charges with dPol polarizabilities.
"""
import json
import logging
import os
from typing import List, Union
import copy

import numpy as np
import ray
from dpolfit.fitting.utils import (
    BCCLocator,
    BCCParameter,
    BondChargeCorrection,
    PolarizabilityParameter,
    compute_am1,
    coulomb_scaling,
    create_dPolmol,
    dPolMol,
    default_bcc_parm,
    default_alpha_parm,
    quality_of_fit,
)
from rdkit import Chem
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def find_bccs(mol: Chem.Mol, bccs: list):
    Chem.Kekulize(mol, clearAromaticFlags=True)
    Chem.SetAromaticity(mol, model=Chem.AromaticityModel.AROMATICITY_MDL)
    ret = []
    for bcc in bccs:
        pt = Chem.MolFromSmarts(bcc)
        if mol.HasSubstructMatch(pt):
            ret.append(BCCLocator(smarts=bcc, indices=mol.GetSubstructMatches(pt)))

    return ret


def assignment_matrix(mol: Chem.Mol, BCCs: List[BCCParameter] = default_bcc_parm):
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
        pattern = match.smarts
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
        raise ValueError(
            f"BCC not fully assigned for {not_assigned} {Chem.MolToSmiles(mol)}"
        )

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


def compute_am1bccdPol(
    rdmol: Chem.Mol,
    BCCs: List[BCCParameter] = default_bcc_parm,
    am1: Union[np.array, None] = None,
    write_sdf: bool = True,
    file_name: str = "molecule.sdf",
) -> Chem.Mol:
    ret = assignment_matrix(mol=rdmol, BCCs=BCCs)
    if am1 is None:
        am1 = compute_am1(rdmol)
    else:
        am1 = am1.reshape(-1)

    am1bccdpol = am1 + ret.applied_correction
    for a in rdmol.GetAtoms():
        a.SetDoubleProp("PartialCharge", am1bccdpol[a.GetIdx()])
    Chem.CreateAtomDoublePropertyList(rdmol, "PartialCharge")
    if write_sdf:
        sdwriter = Chem.SDWriter(file_name)
        sdwriter.write(rdmol)
        sdwriter.close()
    return rdmol


def fit(
    mol: dPolMol,
    BCCs: List[BCCParameter] = default_bcc_parm,
):

    rdmol = mol.rdmol
    n_atoms = rdmol.GetNumAtoms()

    if mol.am1_charges is None:
        precharges = compute_am1(rdmol).reshape(-1)
    else:
        precharges = mol.am1_charges.reshape(-1)

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

    # not sure why there is nan, seems random
    if np.where(np.isnan(drjk))[0].shape[0] > 0:
        logger.info("found nan in drjk")
        drjk = np.nan_to_num(drjk)

    A = np.zeros((n_bccs, n_bccs))
    B = np.zeros(n_bccs)

    matched_bccs = find_bccs(rdmol, bcc_list)
    param_dict = {
        r.smarts: list(set([atom for pair in r.indices for atom in pair]))
        for r in matched_bccs
    }

    esps = mol.data_esp["0"].reshape(-1)
    precharge_esp = np.dot(r_ij1.T, precharges)
    efield = np.zeros((n_atoms, 3))
    for k in range(n_atoms):
        efield[k] = np.dot(precharges, drjk[k])

    deij = np.einsum("jm, jim->ji", efield, r_ij) * r_ij3
    desp = np.dot(mol.polarizabilities, deij)

    vdiff = esps - precharge_esp - desp

    ret_assignment = assignment_matrix(rdmol, BCCs)
    assignment = ret_assignment.assignment

    D = np.zeros((n_bccs, grid.shape[0]))
    for i in range(n_bccs):
        for j in range(n_atoms):
            D[i] += (
                assignment[j, i] * r_ij1[j]
                + np.dot(np.dot(assignment[:, i], drjk[j]), r_ij[j].T)
                * r_ij3[j]
                * mol.polarizabilities[j]
            )

    for k, (smirks_k, bccs_k) in enumerate(param_dict.items()):
        logger.debug(f"Processing {smirks_k}...")

        loc_k = location[smirks_k]
        Cik = D[loc_k]
        B[loc_k] += np.dot(Cik, vdiff).item()

        for l, (smirks_l, bccs_l) in enumerate(param_dict.items()):
            if k > l:
                continue
            loc_l = location[smirks_l]
            Cil = D[loc_l]
            A[loc_k, loc_l] += np.dot(Cil, Cik)

    A += A.T - np.diag(A.diagonal())
    if np.where(np.isnan(B))[0].shape[0] > 0:
        [atom.SetAtomMapNum(0) for atom in rdmol.GetAtoms()]
        rdmol = Chem.RemoveHs(rdmol)
        smi = Chem.MolToSmiles(rdmol, allHsExplicit=False, kekuleSmiles=True)
        print("found nan in B", smi)
        np.savez(f"{smi}.npz", A=A, B=B, D=drjk)
        print("-" * 55)
        return None
    else:
        return A, B


class TrainingSet:
    def __init__(
        self,
        trainingset_path: List,
        BCCs: List[BCCParameter] = default_bcc_parm,
        parameter_list: [PolarizabilityParameter] = default_alpha_parm,
        num_cpus: int = 8,
    ):
        self.trainingset_path = trainingset_path
        self.BCCs = BCCs
        self.parameter_list = parameter_list
        self.num_cpus = num_cpus
        self.results = []
        self.iter = 0
        cwd = os.getcwd()
        if self.num_cpus > 1:
            ray.init(_temp_dir="/tmp/ray_tmp", num_gpus=0, num_cpus=self.num_cpus)
            fit_remote = ray.remote(num_cpus=1, num_gpus=0)(fit)
            self.fit_method = {"fit": fit_remote.remote}
        else:
            self.fit_method = {"fit": fit}
        self.ndim = len(self.BCCs)
        self.location = {p.smarts: idx for idx, p in enumerate(self.BCCs)}

        self.mols = [
            create_dPolmol(data_path=f, parameter_list=self.parameter_list, am1=False)
            for f in self.trainingset_path
            if os.path.exists(os.path.join(f, "grid_esp.0.npy"))
        ]

    def training(self):
        BCCs = copy.deepcopy(self.BCCs)
        A = np.zeros((self.ndim, self.ndim))
        B = np.zeros(self.ndim)
        ret = [self.fit_method["fit"](mol=mol, BCCs=BCCs) for mol in self.mols]
        if self.num_cpus > 1:
            ret = ray.get(ret)
        A += sum([r[0] for r in ret if r is not None])
        B += sum([r[1] for r in ret if r is not None])

        if np.where(B == 0)[0].size > 0:
            not_fitted = np.where(B == 0)[0]

            for i in sorted(not_fitted, reverse=True):
                nft = BCCs.pop(i)
                logger.debug(f"Parameter {nft.smarts} not fitted")
            fitted_smirks = [p.smarts for p in BCCs]
            fitted_positions = [self.location[p] for p in fitted_smirks]

            A = A[tuple(np.meshgrid(fitted_positions, fitted_positions))]
            B = B[fitted_positions]

        else:
            pass

        ret = np.linalg.solve(A, B)
        self.results.append({"iteration": self.iter, "parameters": ret.tolist()})
        new_bccs = [
            BCCParameter(smarts=bcc.smarts, value=value, pid=bcc.pid)
            for bcc, value in zip(BCCs, ret)
        ]

        this_mols = []
        for mol in self.mols:
            charged_mol = compute_am1bccdPol(
                rdmol=mol.rdmol, BCCs=new_bccs, am1=mol.am1_charges, write_sdf=False
            )
            charges = [a.GetDoubleProp("PartialCharge") for a in charged_mol.GetAtoms()]
            mol.am1_charges = np.array(charges)
            mol.rdmol = charged_mol
            this_mols.append(mol)

        rrms = [quality_of_fit(mol, dipole=False)["dpol rrms"] for mol in this_mols]

        ret = [{"smarts": bcc.smarts, "value": value} for bcc, value in zip(BCCs, ret)]
        json.dump(ret, open("results.json", "w"), indent=2)

        return ret

    # def training(self, tol=1e-7):
    #     last_rrms = self._worker()
    #     this_rrms = 0
    #     logger.info(f"RRMS = {last_rrms}")
    #     while abs(last_rrms-this_rrms) > tol:
    #         this_rrms = self._worker()
    #         self.iter += 1
    #         last_rrms = this_rrms
    #         logger.info(f"iteration {self.iter}")
    #         logger.info(f"RRMS = {this_rrms}")
    #     json.dump(self.results, open("results.json", "w"), indent=2)


if __name__ == "__main__":
    print("am1bccdpol module")

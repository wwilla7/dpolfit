#!/usr/bin/env python

"""
Fit polarizability parameters to ESP data
"""

import json
import logging
import os
from typing import List, Dict

import numpy as np
import ray
from dpolfit.fitting.utils import (
    PolarizabilityParameter,
    match_alphas,
    create_dPolmol,
    dPolMol,
)
from scipy.spatial.distance import cdist


logger = logging.getLogger(__name__)


def quality_of_fit(mol: dPolMol) -> Dict[str, float]:
    """
    Returns a dictionary of the
    quality of fit (RRMS) of the polarizability parameters
    """

    crds = mol.coordinates
    grid = mol.grid

    r_ij = -(grid - crds[:, None])
    r_ij3 = np.power(cdist(crds, grid, metric="euclidean"), -3)

    alphas = mol.polarizabilities
    if not alphas.any():
        logger.info("polarizabilities not assigned")

    data_rrms = {}

    for k, v in mol.data_esp.items():
        if k == "0":
            continue

        vdiff = v["vdiff"]
        efield = v["efield"]

        D_ijE = np.dot(r_ij, efield) * r_ij3

        esps = np.einsum("ji,j->i", D_ijE, alphas).reshape(-1, 1)

        rrms = np.sqrt(
            np.sum(np.square(vdiff - esps)) / np.sum(np.square(vdiff)) / len(vdiff)
        )

        data_rrms[k] = rrms

    return data_rrms


def fit(mol: dPolMol, Polarizabilities: List[PolarizabilityParameter]):
    crds = mol.coordinates
    grid = mol.grid
    rdmol = mol.rdmol

    r_ij = -(grid - crds[:, None])
    r_ij3 = np.power(cdist(crds, grid, metric="euclidean"), -3)

    ndim = len(Polarizabilities)

    A = np.zeros((ndim, ndim))
    B = np.zeros(ndim)
    location = {p.smarts: idx for idx, p in enumerate(Polarizabilities)}
    polarizability_list = location.keys()
    matched_alphas = match_alphas(rdmol, polarizability_list)
    param_dict = {
        r.smarts: list(set([atom for pair in r.indices for atom in pair]))
        for r in matched_alphas
    }

    for d, v in mol.data_esp.items():
        if d == "0":
            continue

        logger.debug(f"Processing {d}...")

        vdiff = v["vdiff"]
        efield = v["efield"]

        D_ijE = np.dot(r_ij, efield) * r_ij3

        for k, (smirks_k, alphas_k) in enumerate(param_dict.items()):
            logger.debug(f"Processing {smirks_k}...")

            Cik = D_ijE[alphas_k].sum(axis=0)
            B[location[smirks_k]] += np.dot(Cik, vdiff).item()

            for l, (smirks_l, alphas_l) in enumerate(param_dict.items()):
                if k > l:
                    continue
                Cil = D_ijE[alphas_l].sum(axis=0)
                A[location[smirks_k], location[smirks_l]] += np.dot(Cil, Cik)

    A += A.T - np.diag(A.diagonal())

    return A, B


def training(
    trainingset_path: List,
    Polarizabilities: List[PolarizabilityParameter],
    num_cpus: int = 8,
):
    cwd = os.getcwd()
    fit_method = {}
    if num_cpus > 1:
        ray.init(_temp_dir="/tmp/ray_tmp", num_gpus=0, num_cpus=num_cpus)
        fit_remote = ray.remote(num_cpus=1, num_gpus=0)(fit)
        # quality_of_fit_remote = ray.remote(num_cpus=1, num_gpus=0)(quality_of_fit)

        fit_method["fit"] = fit_remote.remote
        # fit_method["quality_of_fit"] = quality_of_fit_remote.remote
    else:
        fit_method["fit"] = fit
        # fit_method["quality_of_fit"] = quality_of_fit

    ndim = len(Polarizabilities)
    A = np.zeros((ndim, ndim))
    B = np.zeros(ndim)
    location = {p.smarts: idx for idx, p in enumerate(Polarizabilities)}
    mols = [create_dPolmol(data_path=f, parameter_list=None) for f in trainingset_path]
    ret = [
        fit_method["fit"](mol=mol, Polarizabilities=Polarizabilities) for mol in mols
    ]
    if num_cpus > 1:
        ret = ray.get(ret)
    A += sum([r[0] for r in ret if r is not None])
    B += sum([r[1] for r in ret if r is not None])

    if np.where(B == 0)[0].size > 0:
        not_fitted = np.where(B == 0)[0]

        for i in sorted(not_fitted, reverse=True):
            nft = Polarizabilities.pop(i)
            logger.info(f"Parameter {nft.smarts} not fitted")

        fitted_smirks = [p.smarts for p in Polarizabilities]
        fitted_positions = [location[p] for p in fitted_smirks]

        A = A[tuple(np.meshgrid(fitted_positions, fitted_positions))]
        B = B[fitted_positions]

    else:
        pass

    np.savez(os.path.join(cwd, "matrices.npz"), A=A, B=B)

    logger.info(f"A: {A}")
    logger.info(f"B: {B}")

    ret = np.linalg.solve(A, B)

    results = []
    for p, v in zip(Polarizabilities, ret):
        p.value = v
        results.append(p.__dict__)

    json.dump(results, open(os.path.join(cwd, "results.json"), "w"), indent=2)
    return Polarizabilities

    # # # check rrms
    # all_rrms = []

    # new_mols = [
    #     create_dPolmol(data_path=f, parameter_list=Polarizabilities)
    #     for f in trainingset_path
    #     if os.path.exists(os.path.join(f, "grid_esp.0.npy"))
    # ]

    # rrms = [fit_method["quality_of_fit"](mol=mol) for mol in new_mols]
    # if num_cpus > 1:
    #     rrms = ray.get(rrms)
    # all_rrms.extend([list(r.values()) for r in rrms])

    # logger.info(f"RRMS: {np.mean(all_rrms):.5f}")
    # return rrms


if __name__ == "__main__":
    print("Module to fit polarizabilities")

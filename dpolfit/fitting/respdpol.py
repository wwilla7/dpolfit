#!/usr/bin/env python

"""
Fit partial charges for typed polarizabilities
"""

import copy
import os
from typing import Dict, List, Optional, Tuple
from dpolfit.fitting.utils import dPolMol, coulomb_scaling, pair_equivalent
import numpy as np
import pandas as pd
import pint
from rdkit import Chem
from scipy.spatial.distance import cdist

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
import logging

logger = logging.getLogger(__name__)


def calc_desp(
    mol: dPolMol,
    qs: np.ndarray,
    alphas: List,
    drjk: np.ndarray,
    r_ij: np.ndarray,
    r_ij3: np.ndarray,
) -> np.ndarray:
    """
    Calculate the ESP from induced dipoles of the molecule.
    Unit: e / bohr
    """

    natom = len(mol.coordinates)

    efield = np.zeros((natom, 3))
    for k in range(natom):
        efield[k] = np.dot(qs, drjk[k])

    deij = np.einsum("jm, jim->ji", efield, r_ij) * r_ij3.T

    desp = np.dot(alphas, deij)

    return desp


def fit(mol: dPolMol) -> np.ndarray:
    """
    Fit RESP-style charges to baseline ESPs

    Parameters
    ----------
    mol : Mol

    Returns
    -------
    respp : np.ndarray
    respdpol2 : np.ndarray
    dpol_rrms : float
    """

    crds = mol.coordinates
    grid = mol.grid
    esps = mol.data_esp.reshape(-1)
    offmol = mol.offmol
    alphas = mol.polarizabilities
    natoms = len(crds)
    npoints = len(grid)

    r_ij = -(grid - crds[:, None])  # distance vector of grid points from atoms
    r_ij0 = cdist(grid, crds, metric="euclidean")
    r_ij1 = np.power(r_ij0, -1)  # euclidean distance of atoms and grids ^ -1
    r_ij3 = np.power(r_ij0, -3)  # euclidean distance of atoms and grids ^ -3

    r_jk = crds - crds[:, None]  # distance vector of atoms from each other
    r_jk1 = cdist(
        crds, crds, metric="euclidean"
    )  # euclidean distance of atoms from each other
    r_jk3 = np.power(
        r_jk1, -3, where=r_jk1 != 0
    )  # euclidean distance of atoms from each other ^ -3

    rdmol = offmol.to_rdkit()
    chemically_equivalent_atoms = list(
        Chem.rdmolfiles.CanonicalRankAtoms(rdmol, breakTies=False)
    )
    chemically_equivalent_atoms_pairs = pair_equivalent(chemically_equivalent_atoms)
    n_chemically_equivalent_atoms = len(chemically_equivalent_atoms_pairs)
    net_charge = offmol.total_charge.m_as(unit.elementary_charge)
    coulomb14scale_matrix = coulomb_scaling(rdmol, coulomb14scale=0.5)
    forced_symmetry = set(
        [item for sublist in chemically_equivalent_atoms_pairs for item in sublist]
    )
    polar_region = list(set(range(natoms)) - forced_symmetry)
    n_polar_region = len(polar_region)
    elements = [atom.GetSymbol() for atom in rdmol.GetAtoms()]

    # Distance dependent matrices for fitting

    drjk = np.zeros((natoms, natoms, 3))
    for k in range(natoms):
        drjk[k] = r_jk[k] * (r_jk3[k] * coulomb14scale_matrix[k]).reshape(-1, 1)

    # start charge-fitting
    # first stage, no symmetry
    ndim1 = natoms + 1
    a0 = np.einsum("ij,ik->jk", r_ij1, r_ij1)
    a1 = np.zeros((ndim1, ndim1))
    a1[:natoms, :natoms] = a0

    # Lagrange multiplier
    a1[natoms, :] = 1.0
    a1[:, natoms] = 1.0
    a1[natoms, natoms] = 0.0

    b1 = np.zeros(ndim1)
    b1[:natoms] = np.einsum("ik,i->k", r_ij1, esps)
    b1[natoms] = net_charge

    q1 = np.linalg.solve(a1, b1)[:natoms]

    q11 = np.zeros(natoms)

    while not np.allclose(q1, q11):
        a10 = copy.deepcopy(a1)
        for j in range(natoms):
            if elements[j] != "H":
                a10[j, j] += 0.0005 * np.power((q1[j] ** 2 + 0.1**2), -0.5)
        q1 = q11
        q11 = np.linalg.solve(a10, b1)[:natoms]

    resp1 = q11

    # second stage, apply forced symmetry
    ndim2 = natoms + 1 + n_chemically_equivalent_atoms + n_polar_region
    a2 = np.zeros((ndim2, ndim2))
    a2[: natoms + 1, : natoms + 1] = a1

    if n_chemically_equivalent_atoms == 0:
        pass
    else:
        for idx, pair in enumerate(chemically_equivalent_atoms_pairs):
            a2[natoms + 1 + idx, pair[0]] = 1.0
            a2[natoms + 1 + idx, pair[1]] = -1.0
            a2[pair[0], natoms + 1 + idx] = 1.0
            a2[pair[1], natoms + 1 + idx] = -1.0

    b2 = np.zeros(ndim2)
    b2[natoms] = net_charge

    charge_to_be_fixed = q1[polar_region]

    for idx, pol_idx in enumerate(polar_region):
        a2[ndim2 - n_polar_region + idx, pol_idx] = 1.0
        a2[pol_idx, ndim2 - n_polar_region + idx] = 1.0
        b2[ndim2 - n_polar_region + idx] = charge_to_be_fixed[idx]

    q2 = resp1
    q22 = np.zeros(natoms)
    steps = 0
    while not np.allclose(q2, q22) and steps < 20:
        a20 = copy.deepcopy(a2)
        for j in range(natoms):
            if elements[j] != "H":
                a20[j, j] += 0.001 * np.power((q2[j] ** 2 + 0.1**2), -0.5)

        desp = calc_desp(
            mol=mol, qs=q2, alphas=alphas, drjk=drjk, r_ij=r_ij, r_ij3=r_ij3
        )
        esp_to_fit = esps - desp
        b2[:natoms] = np.einsum("ik,i->k", r_ij1, esp_to_fit)
        q2 = q22
        q22 = np.linalg.solve(a20, b2)[:natoms]
        steps += 1

    resp2 = q22

    # quality of fit
    base_esp = np.dot(r_ij1, resp2)
    dpol_esp = calc_desp(
        mol=mol, qs=q2, alphas=alphas, drjk=drjk, r_ij=r_ij, r_ij3=r_ij3
    )
    final_esp = base_esp + dpol_esp

    ## rrms
    y = lambda x: np.sqrt((sum((x - esps) ** 2) / sum(esps**2)) / npoints)

    dpol_rrms = y(final_esp)

    return resp1, resp2, dpol_rrms


if __name__ == "__main__":
    import json
    from glob import glob

    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data")
    ff_path = os.path.join(cwd, "custom.offxml")
    pol_handler = "MPIDPolarizability"
    molecules = glob(os.path.join(data_path, "mol*"))
    nmol = len(molecules)

    if pol_handler == "MPIDPolarizability":
        from mpid_plugin.nonbonded import (
            MPIDCollection,
            MPIDMultipoleHandler,
            MPIDPolarizabilityHandler,
        )

        ff = ForceField(ff_path, load_plugins=True, allow_cosmetic_attributes=True)
    else:
        ff = ForceField(ff_path)

    data_dict = []

    for idx, mol in enumerate(molecules):
        logger.info(f"Fitting molecule {idx+1}/{nmol}")
        if os.path.exists(os.path.join(mol, "conf0")):
            confs = glob(os.path.join(mol, "conf*"))
        else:
            confs = [mol]

        for conf in confs:
            try:
                this_mol = Mol(conf, ff, pol_handler)
                this_offmol = this_mol.offmol
                respp, respdpol, rrms = fit(this_mol)
                this_offmol.partial_charges = respdpol * unit.elementary_charge
                this_offmol.to_file(
                    os.path.join(conf, "mol_charged.sdf"), file_format="sdf"
                )

                data_dict.append(
                    {
                        "mapped_smiles": this_mol.mapped_smiles,
                        "respp": respp.tolist(),
                        "respdpol": respdpol.tolist(),
                        "rrms": rrms,
                        "pols": this_mol.polarizabilities,
                        "path": mol,
                    }
                )
            except Exception as e:
                print(mol, e)

    json.dump(data_dict, open(os.path.join(cwd, "ret.json"), "w"), indent=2)

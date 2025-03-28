#!/usr/bin/env python

"""
Fit polarizability parameters to ESP data
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pint
from scipy.spatial.distance import cdist

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity
import logging

from openff.toolkit import ForceField, Molecule, Topology
from openff.units import unit

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)
logger = logging.getLogger()


@dataclass
class AMol:
    """
    A molecule with ESP data
    """

    data_path: str
    off_ff: ForceField
    pol_handler: str = "MPIDPolarizability"

    @property
    def mapped_smiles(self) -> str:
        """
        Returns the mapped smiles string of the molecule
        """

        with open(os.path.join(self.data_path, "molecule.smi"), "r") as f:
            smi = f.read()
        return smi

    @property
    def coordinates(self) -> np.ndarray:
        """
        Returns the coordinates of the molecule
        Units: bohr
        """

        crds = (
            Q_(
                np.load(os.path.join(self.data_path, "coordinates.npy")),
                ureg.angstrom,
            )
            .to(ureg.bohr)
            .magnitude
        )
        return crds

    @property
    def grid(self) -> np.ndarray:
        """
        Returns the grid points
        Units: bohr
        """

        grid = (
            Q_(np.load(os.path.join(self.data_path, "grid.npy")), ureg.angstrom)
            .to(ureg.bohr)
            .magnitude
        )

        return grid

    @property
    def data_esp(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Returns all ESP data for polarizability fitting
        Units: e / bohr
        """

        data = {}

        grid_espi_0 = Q_(
            np.load(os.path.join(self.data_path, "grid_esp.0.npy")),
            ureg.elementary_charge / ureg.bohr,
        ).magnitude

        data["0"] = grid_espi_0

        perturb_dipoles = {
            "x+": [0.01, 0.0, 0.0],
            "x-": [-0.01, 0.0, 0.0],
            "y+": [0.0, 0.01, 0.0],
            "y-": [0.0, -0.01, 0.0],
            "z+": [0.0, 0.0, 0.01],
            "z-": [0.0, 0.0, -0.01],
        }

        for d, e in perturb_dipoles.items():
            grid_espi = Q_(
                np.load(os.path.join(self.data_path, f"grid_esp.{d}.npy")),
                ureg.elementary_charge / ureg.bohr,
            ).magnitude
            vdiffi = grid_espi - grid_espi_0
            data[d] = {"vdiff": vdiffi, "efield": e}

        return data

    @property
    def polarizability_smirks(self) -> List[str]:
        """
        Returns the SMARTS patterns of the polarizability parameters
        """
        mol = Molecule.from_mapped_smiles(self.mapped_smiles)
        parameters = self.off_ff.label_molecules(mol.to_topology())[0]
        smirks = [v._smirks for _, v in parameters[self.pol_handler].items()]
        logger.info(f"smiles: {mol.to_smiles(explicit_hydrogens=False)}")
        return smirks


def quality_of_fit(mol: AMol) -> Dict[str, float]:
    """
    Returns a dictionary of the
    quality of fit (RRMS) of the polarizability parameters
    """

    crds = mol.coordinates
    grid = mol.grid

    r_ij = -(grid - crds[:, None])
    r_ij3 = np.power(cdist(crds, grid, metric="euclidean"), -3)

    offmol = Molecule.from_mapped_smiles(mol.mapped_smiles)
    parameters = mol.off_ff.label_molecules(offmol.to_topology())[0]
    alphas = np.array(
        [
            Q_(v.epsilon.magnitude, "angstrom**3").to("a0**3").magnitude
            # v.polarizabilityXX.to("a0**3").magnitude
            for _, v in parameters[mol.pol_handler].items()
        ]
    )

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


def fit(mol: AMol, ndim: int, positions: Dict[str, int]):
    crds = mol.coordinates
    grid = mol.grid

    r_ij = -(grid - crds[:, None])
    r_ij3 = np.power(cdist(crds, grid, metric="euclidean"), -3)

    r_jk = crds - crds[:, None]
    r_jk1 = cdist(crds, crds, metric="euclidean")
    r_jk3 = np.power(r_jk1, -3, where=r_jk1 != 0)

    A = np.zeros((ndim, ndim))
    B = np.zeros(ndim)

    param_dict = defaultdict(list)
    for idx, smirks in enumerate(mol.polarizability_smirks):
        param_dict[smirks].append(idx)

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
            B[positions[smirks_k]] += np.dot(Cik, vdiff).item()

            for l, (smirks_l, alphas_l) in enumerate(param_dict.items()):
                if k > l:
                    continue
                Cil = D_ijE[alphas_l].sum(axis=0)
                A[positions[smirks_k], positions[smirks_l]] += np.dot(Cil, Cik)

    A += A.T - np.diag(A.diagonal())

    return A, B


if __name__ == "__main__":
    from glob import glob

    # from mpid_plugin.nonbonded import MPIDCollection, MPIDPolarizabilityHandler

    a03_to_ang3 = Q_(1, "bohr**3").to("angstrom**3").magnitude

    cwd = os.getcwd()
    data_path = os.path.join(cwd, "data")
    ff = ForceField(os.path.join(cwd, "input.offxml"))  # , load_plugins=True)
    pol_handler = "vdW"
    # pol_handler = "MPIDPolarizability"

    parameters = ff.get_parameter_handler(pol_handler).parameters
    param_positions = {p._smirks: i for i, p in enumerate(parameters)}

    ndim = len(parameters)
    A = np.zeros((ndim, ndim))
    B = np.zeros(ndim)

    molecules = glob(os.path.join(data_path, "molecule*"))
    nmol = len(molecules)
    for idx, mol in enumerate(molecules):
        logger.info(f"Fitting molecule {idx+1}/{nmol}")
        confs = glob(os.path.join(mol, "conf*"))
        amols = [AMol(conf, ff, pol_handler) for conf in confs]
        ret = [fit(amol, ndim, param_positions) for amol in amols]

        A += sum([r[0] for r in ret])
        B += sum([r[1] for r in ret])

    if np.where(B == 0)[0].size > 0:
        not_fitted = np.where(B == 0)[0]

        for i in sorted(not_fitted, reverse=True):
            nft = parameters.pop(i)
            logger.info(f"Parameter {nft._smirks} not fitted")

        fitted_smirks = [p._smirks for p in parameters]
        fitted_positions = [param_positions[p] for p in fitted_smirks]

        A = A[tuple(np.meshgrid(fitted_positions, fitted_positions))]
        B = B[fitted_positions]

    else:
        pass

    np.save(os.path.join(cwd, "A.npy"), A)
    np.save(os.path.join(cwd, "B.npy"), B)

    logger.debug(f"A: {A}")
    logger.debug(f"B: {B}")

    ret = np.linalg.solve(A, B)

    for p, v in zip(parameters, ret):
        pol = v * a03_to_ang3
        p.epsilon = pol * unit.kilocalorie_per_mole
        p.id = f"polarizability = {pol:.5f} * angstrom**3"

    logger.info(f"Saving polarzabilities in the parameter field of `id`.")
    ff.to_file(os.path.join(cwd, "output.offxml"))

    ## check rrms

    all_rrms = []

    for idx, mol in enumerate(molecules):
        confs = glob(os.path.join(mol, "conf*"))
        amols = [AMol(conf, ff, pol_handler) for conf in confs]
        rrms_this_mol = [quality_of_fit(mol=amol) for amol in amols]
        all_rrms.extend([list(r.values()) for r in rrms_this_mol])

    logger.info(f"RRMS: {np.mean(all_rrms):.5f}")

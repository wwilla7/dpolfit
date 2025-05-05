#!/usr/bin/env python
"""
Useful functions to fit partial charges and polarizabilities.
"""
import importlib.resources
import json
import logging
import os
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np
import pint
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

bohr_2_angstrom = 0.529177

perturb_dipoles = {
    "x+": [0.01, 0.0, 0.0],
    "x-": [-0.01, 0.0, 0.0],
    "y+": [0.0, 0.01, 0.0],
    "y-": [0.0, -0.01, 0.0],
    "z+": [0.0, 0.0, 0.01],
    "z-": [0.0, 0.0, -0.01],
}


@dataclass
class dPolMol:
    mapped_smiles: str
    coordinates: np.ndarray
    grid: np.ndarray
    data_esp: Dict[str, Union[np.ndarray, Dict]]
    partial_charges: np.ndarray
    am1_charges: Union[np.ndarray, None]
    polarizabilities: np.ndarray
    rdmol: Chem.Mol


@dataclass
class PolarizabilityParameter:
    smarts: str
    value: float
    pid: int


@dataclass
class PolarizabilityLocator:
    smarts: str
    indices: tuple


@dataclass
class BCCParameter:
    smarts: str
    value: float
    pid: int


@dataclass
class BCCLocator:
    smarts: str
    indices: tuple


@dataclass
class BondChargeCorrection:
    assignment: np.array
    correction: np.array
    applied_correction: np.array


_default_bcc_path = os.path.join(
    importlib.resources.files("dpolfit"), "data", "dpol_bccs.json"
)
_default_bcc_data = json.load(open(_default_bcc_path, "r"))
default_bcc_parm = [
    BCCParameter(smarts=r["smarts"], value=r["value"], pid=idx)
    for idx, r in enumerate(_default_bcc_data)
]

_default_alpha_path = os.path.join(
    importlib.resources.files("dpolfit"), "data", "dpol_alphas.json"
)
_default_alpha_data = json.load(open(_default_alpha_path, "r"))
default_alpha_parm = [
    PolarizabilityParameter(smarts=r["smarts"], value=r["value"], pid=idx)
    for idx, r in enumerate(_default_alpha_data)
]


def pair_equivalent(pattern: List) -> np.ndarray:
    """
    A function to pair related patterns together for use as constraints

    Parameters
    ----------
    pattern: List
        A list of patterns, could be elements, SMIRNOFF patterns

    Returns
    -------
    ndarry
        Return pairs of related patterns in a nested numpy ndarry.

    """
    tmp1 = defaultdict(list)
    for idx1, p in enumerate(pattern):
        tmp1[p].append(idx1)

    tmp2 = []
    for key, v in tmp1.items():
        n = len(v)
        if n > 1:
            tmp2.append([[v[i], v[i + 1]] for i in range(n - 1)])
    if len(tmp2) == 0:
        ret = []
    else:
        ret = np.concatenate(tmp2)
    return ret


def coulomb_scaling(rdmol: Chem.rdchem.Mol, coulomb14scale: float = 0.5) -> np.ndarray:
    """

    Parameters
    ----------
    rdmol: Chem.rdchem.Mol
        An input rdkit molecule used for specifying connectivity

    coulomb14scale: float

    Returns
    -------
    ndarray

    """

    natom = rdmol.GetNumAtoms()
    # initializing arrays
    bonds = []
    bound12 = np.zeros((natom, natom))
    bound13 = np.zeros((natom, natom))
    scaling_matrix = np.ones((natom, natom))

    for bond in rdmol.GetBonds():
        b = bond.GetBeginAtomIdx()
        e = bond.GetEndAtomIdx()
        bonds.append([b, e])

    # Find 1-2 scaling_matrix
    for pair in bonds:
        bound12[pair[0], pair[1]] = 12.0
        bound12[pair[1], pair[0]] = 12.0

    # Find 1-3 scaling_matrix
    b13_pairs = []
    for i in range(natom):
        b12_idx = np.nonzero(bound12[i])[0]
        for idx, j in enumerate(b12_idx):
            for k in b12_idx[idx + 1 :]:
                b13_pairs.append([j, k])
    for pair in b13_pairs:
        bound13[pair[0], pair[1]] = 13.0
        bound13[pair[1], pair[0]] = 13.0

    # Find 1-4 scaling_matrix
    b14_pairs = []
    for i in range(natom):
        b12_idx = np.nonzero(bound12[i])[0]
        for j in b12_idx:
            b122_idx = np.nonzero(bound12[j])[0]
            for k in b122_idx:
                for j2 in b12_idx:
                    if k != i and j2 != j:
                        b14_pairs.append([j2, k])

    # Assign coulomb14scaling factor
    for pair in b14_pairs:
        scaling_matrix[pair[0], pair[1]] = coulomb14scale
        scaling_matrix[pair[1], pair[0]] = coulomb14scale

    # Exclude 1-2, 1-3 interactions
    for pair in bonds:
        scaling_matrix[pair[0], pair[1]] = 0.0
        scaling_matrix[pair[1], pair[0]] = 0.0

    for pair in b13_pairs:
        scaling_matrix[pair[0], pair[1]] = 0.0
        scaling_matrix[pair[1], pair[0]] = 0.0

    # Fill 1-1 with zeros
    np.fill_diagonal(scaling_matrix, 0)

    return scaling_matrix


def compute_am1(rdmol: Chem.Mol, delete_files=True, am1bcc=False):
    n_atoms = rdmol.GetNumAtoms()
    formal_charge = Chem.GetFormalCharge(rdmol)
    cwd = os.getcwd()
    tmp_path = os.path.join(cwd, str(uuid.uuid4()))
    os.makedirs(tmp_path)
    os.chdir(tmp_path)
    AllChem.MMFFOptimizeMolecule(rdmol)
    conf = rdmol.GetConformer()
    conf.Set3D(True)
    sd = Chem.SDWriter("molecule.sdf")
    sd.write(rdmol)
    sd.close()
    if am1bcc:
        charge_method = "bcc"
    else:
        charge_method = "mul"
    output = subprocess.run(
        [
            "antechamber",
            "-i",
            "molecule.sdf",
            "-fi",
            "sdf",
            "-o",
            "molecule.mol2",
            "-fo",
            "mol2",
            "-c",
            charge_method,
            "-nc",
            str(formal_charge),
            "-at",
            "sybyl",
            "-dr",
            "no",
        ],
        capture_output=True,
    )
    output = subprocess.run(
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
            "-dr",
            "no",
        ],
        capture_output=True,
    )
    if os.path.exists("charges.dat"):
        with open("charges.dat", "r") as f:
            charges = f.read().split()
        charges = np.array(charges).astype(float)
        charges += (formal_charge - charges.sum()) / n_atoms
        os.chdir(cwd)
        if delete_files:
            shutil.rmtree(tmp_path)
        return charges
    else:
        smi = Chem.MolToSmiles(rdmol)
        logger.info(smi)
        os.chdir(cwd)
        raise FileNotFoundError(tmp_path)


def match_alphas(rdmol: Chem.Mol, alpha_list: List):
    Chem.Kekulize(rdmol, clearAromaticFlags=True)
    Chem.SetAromaticity(rdmol, model=Chem.AromaticityModel.AROMATICITY_MDL)
    ret = []
    for alpha in alpha_list:
        pt = Chem.MolFromSmarts(alpha)
        if rdmol.HasSubstructMatch(pt):
            ret.append(
                PolarizabilityLocator(
                    smarts=alpha, indices=rdmol.GetSubstructMatches(pt)
                )
            )

    return ret


def create_dPolmol(
    data_path: str,
    parameter_list: Union[List[PolarizabilityParameter], None] = None,
    am1: bool = False,
):
    smi_path = os.path.join(data_path, "molecule.smi")
    mol_path = os.path.join(data_path, "optimized.sdf")
    if os.path.exists(smi_path):
        with open(smi_path, "r") as f:
            smi = f.read().split("\n")[0]
        rdmol = Chem.MolFromSmiles(smi)
        rdmol = Chem.AddHs(rdmol)
    else:
        sdreader = Chem.SDMolSupplier(mol_path, removeHs=False)
        rdmol = [m for m in sdreader][0]
        for a in rdmol.GetAtoms():
            a.SetAtomMapNum(a.GetIdx() + 1)
        smi = Chem.MolToSmiles(rdmol, allHsExplicit=True)

    Chem.Kekulize(rdmol, clearAromaticFlags=True)
    Chem.SetAromaticity(rdmol, model=Chem.AromaticityModel.AROMATICITY_MDL)

    crds = (
        Q_(
            np.load(os.path.join(data_path, "coordinates.npy")),
            ureg.angstrom,
        )
        .to(ureg.bohr)
        .magnitude
    )

    n_atoms = rdmol.GetNumAtoms()
    AllChem.Compute2DCoords(rdmol)
    conf = rdmol.GetConformer()
    for i in range(rdmol.GetNumAtoms()):
        xyz = crds[i] * bohr_2_angstrom
        conf.SetAtomPosition(i, Point3D(*xyz))

    grid = (
        Q_(np.load(os.path.join(data_path, "grid.npy")), ureg.angstrom)
        .to(ureg.bohr)
        .magnitude
    )

    data = {}

    grid_espi_0 = Q_(
        np.load(os.path.join(data_path, "grid_esp.0.npy")),
        ureg.elementary_charge / ureg.bohr,
    ).magnitude

    data["0"] = grid_espi_0

    for d, e in perturb_dipoles.items():
        this_esp = os.path.join(data_path, f"grid_esp.{d}.npy")
        if os.path.exists(this_esp):
            grid_espi = Q_(
                np.load(this_esp), ureg.elementary_charge / ureg.bohr
            ).magnitude
            vdiffi = grid_espi - grid_espi_0
            data[d] = {"vdiff": vdiffi, "efield": e}

    if isinstance(parameter_list, List):
        alpha_dict = {p.smarts: p.value for p in parameter_list}
        alpha_list = alpha_dict.keys()
        matched_alphas = match_alphas(rdmol=rdmol, alpha_list=alpha_list)
        alpha_values = np.zeros(n_atoms)
        for match in matched_alphas:
            for idx in match.indices:
                v = alpha_dict[match.smarts]
                alpha_values[idx] = v
    else:
        alpha_values = np.zeros(n_atoms)
        logger.info("polarizabilities not found")

    if am1:
        am1_charges = compute_am1(rdmol)
    else:
        am1_charges = None

    ret = dPolMol(
        mapped_smiles=smi,
        coordinates=crds,
        grid=grid,
        data_esp=data,
        partial_charges=np.zeros(n_atoms),
        am1_charges=am1_charges,
        polarizabilities=alpha_values,
        rdmol=rdmol,
    )
    return ret


def quality_of_fit(mol: dPolMol, dipole=False):

    rdmol = mol.rdmol
    n_atoms = rdmol.GetNumAtoms()
    esps = mol.data_esp["0"].reshape(-1)
    permanent_charges = np.array(
        [a.GetDoubleProp("PartialCharge") for a in rdmol.GetAtoms()]
    )

    crds = mol.coordinates
    grid = mol.grid
    npoints = grid.shape[0]

    r_ij = -(grid - crds[:, None])
    r_ij0 = cdist(crds, grid, metric="euclidean")
    r_ij1 = np.power(r_ij0, -1)
    r_ij3 = np.power(r_ij0, -3)

    r_jk = crds - crds[:, None]
    r_jk1 = cdist(crds, crds, metric="euclidean")
    r_jk3 = np.power(r_jk1, -3, where=r_jk1 != 0)

    coulomb14scale_matrix = coulomb_scaling(rdmol, coulomb14scale=0.5)
    # coulomb14scale_matrix = coulomb_scaling(rdmol, coulomb14scale=0.83333)

    drjk = np.zeros((n_atoms, n_atoms, 3))
    for k in range(n_atoms):
        drjk[k] = r_jk[k] * (r_jk3[k] * coulomb14scale_matrix[k]).reshape(-1, 1)

    # not sure why there is nan, seems random
    if np.where(np.isnan(drjk))[0].shape[0] > 0:
        logger.info("found nan in drjk")
        drjk = np.nan_to_num(drjk)

    permanent_esp = np.dot(r_ij1.T, permanent_charges)

    efield = np.zeros((n_atoms, 3))
    for k in range(n_atoms):
        efield[k] = np.dot(permanent_charges, drjk[k])

    deij = np.einsum("jm, jim->ji", efield, r_ij) * r_ij3
    desp = np.dot(mol.polarizabilities, deij)

    calc_esps = desp + permanent_esp

    ## rrms
    y = lambda x: np.sqrt((sum((x - esps) ** 2) / sum(esps**2)) / npoints)

    dpol_rrms = y(calc_esps)
    am1_rrms = y(permanent_esp)
    ret = {"dpol rrms": dpol_rrms, "permanent rrms": am1_rrms}

    if dipole:
        permanent_dipole = np.dot(
            crds.transpose(1, 0), permanent_charges.reshape(-1, 1)
        ).reshape(-1)
        induced = np.dot(
            efield.transpose(1, 0), mol.polarizabilities.reshape(-1, 1)
        ).reshape(-1)
        # Multiply by 2.5417464519 to convert [e a0] to [Debye]
        total = np.linalg.norm(permanent_dipole + induced) * 2.5417464519
        ret |= {
            "total dipole": total,
            "permanent dipole": np.linalg.norm(permanent_dipole) * 2.5417464519,
        }

    return ret


def compute_dipole(
    charged_rdmol: Chem.Mol,
    parameter_list: List[PolarizabilityParameter] = default_alpha_parm,
):
    permanent_charges = np.array(
        [a.GetDoubleProp("PartialCharge") for a in charged_rdmol.GetAtoms()]
    )
    conf = charged_rdmol.GetConformer()
    crds = conf.GetPositions() / bohr_2_angstrom
    n_atoms = charged_rdmol.GetNumAtoms()

    alpha_dict = {p.smarts: p.value for p in parameter_list}
    alpha_list = alpha_dict.keys()
    matched_alphas = match_alphas(rdmol=charged_rdmol, alpha_list=alpha_list)
    alpha_values = np.zeros(n_atoms)
    for match in matched_alphas:
        for idx in match.indices:
            v = alpha_dict[match.smarts]
            alpha_values[idx] = v

    r_jk = crds - crds[:, None]
    r_jk1 = cdist(crds, crds, metric="euclidean")
    r_jk3 = np.power(r_jk1, -3, where=r_jk1 != 0)

    coulomb14scale_matrix = coulomb_scaling(charged_rdmol, coulomb14scale=0.5)

    drjk = np.zeros((n_atoms, n_atoms, 3))
    for k in range(n_atoms):
        drjk[k] = r_jk[k] * (r_jk3[k] * coulomb14scale_matrix[k]).reshape(-1, 1)

    # not sure why there is nan, seems random
    if np.where(np.isnan(drjk))[0].shape[0] > 0:
        logger.info("found nan in drjk")
        drjk = np.nan_to_num(drjk)

    efield = np.zeros((n_atoms, 3))
    for k in range(n_atoms):
        efield[k] = np.dot(permanent_charges, drjk[k])

    permanent_dipole = np.dot(
        crds.transpose(1, 0), permanent_charges.reshape(-1, 1)
    ).reshape(-1)
    induced = np.dot(efield.transpose(1, 0), alpha_values.reshape(-1, 1)).reshape(-1)
    # Multiply by 2.5417464519 to convert [e a0] to [Debye]
    total = np.linalg.norm(permanent_dipole + induced) * 2.5417464519
    ret = {
        "total dipole": total,
        "permanent dipole": np.linalg.norm(permanent_dipole) * 2.5417464519,
    }

    return ret

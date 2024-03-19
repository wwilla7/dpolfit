#!/usr/bin/env python

"""
Fit partial charges for typed polarizabilities
"""

import copy
from collections import defaultdict

import json
import numpy as np
import pint
from rdkit import Chem
from scipy.spatial.distance import cdist

try:
    from openeye import oechem
    from oechem import OEField, Types
    from dpolfit.utilities.miscellaneous import *
except ModuleNotFoundError:
    from dpolfit.utilities import oechem
    smiTags = None

    print("Don't have openeye toolkit installed")

from dpolfit.fitting.polarizability import label_alpha

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def pair_equivalent(pattern: list) -> np.ndarray:
    """
    A function to pair related patterns together for use as constraints

    Parameters
    ----------
    pattern: list
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


def coulomb_scaling_rdkit(
    rdmol: Chem.rdchem.Mol, coulomb14scale: float = 0.5
) -> np.ndarray:
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


def coulomb_scaling_oe(oemol: oechem.OEMol, coulomb14scale: float = 0.5) -> np.ndarray:

    natom = oemol.GetMaxAtomIdx()
    # initializing arrays
    bonds = []
    bound12 = np.zeros((natom, natom))
    bound13 = np.zeros((natom, natom))
    scaling_matrix = np.ones((natom, natom))

    for bond in oemol.GetBonds():
        b = bond.GetBgnIdx()
        e = bond.GetEndIdx()
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


def calc_desp(
    natom: int,
    qs: np.ndarray,
    alphas: list,
    drjk: np.ndarray,
    r_ij: np.ndarray,
    r_ij3: np.ndarray,
) -> np.ndarray:
    """
    Calculate the ESP from induced dipoles of the molecule.
    Unit: e / bohr
    """

    efield = np.zeros((natom, 3))
    for k in range(natom):
        efield[k] = np.dot(qs, drjk[k])

    deij = np.einsum("jm, jim->ji", efield, r_ij) * r_ij3.T

    desp = np.dot(alphas, deij)

    return desp


def generate_paired_parameters(oemol: oechem.OEMol, tags=smiTags) -> dict:
    ret = {}
    # clear potential mapped index
    [atom.SetMapIdx(0) for atom in oemol.GetAtoms()]
    for atom in oemol.GetAtoms():
        idx = atom.GetIdx()
        atom.SetMapIdx(idx + 1)
        pattern = oechem.OECreateSmiString(oemol, tags)
        atom.SetMapIdx(0)
        ret[pattern] = atom.GetData("polarizability")

    return ret


def assign_polarizability(
    oemol: oechem.OEMol, polarizabilities: dict, unit: str
) -> oechem.OEMol:

    conversion = Q_(1, unit).to("bohr**3").magnitude

    # hierarchy
    included = []

    for smarts, value in polarizabilities.items():
        labelled = label_alpha(oemol=oemol, smarts_pattern=smarts, index=False)

        if len(labelled) > 0:
            for atom in labelled:
                atom_idx = atom.GetIdx()
                if atom_idx in included:
                    pass
                else:
                    included.append(atom_idx)
                    atom.SetData("polarizability", value * conversion)

    oemol.SetData("polarizability unit", "bohr**3")

    return oemol


def fit(oerecord: oechem.OEMolRecord, polarizabilities: dict, unit: str) -> np.ndarray:

    oemol = oerecord.get_mol()
    natom = oemol.GetMaxAtomIdx()
    oemol = assign_polarizability(oemol, polarizabilities, unit)
    alphas = [a.GetData("polarizability") for a in oemol.GetAtoms()]  # bohr**3

    for conf in oemol.GetConfs():
        conf_record = oerecord.get_conf_record(conf)
        geometry_angstrom = json.loads(
            conf_record.get_value(OEField("geometry_angstrom", Types.String))
        )

        geometry_bohr = Q_(geometry_angstrom, ureg.angstrom).to(ureg.bohr).magnitude

        grid_angstrom = json.loads(
            conf_record.get_value(OEField("grid_angstrom", Types.String))
        )

        grid_bohr = Q_(grid_angstrom, ureg.angstrom).to(ureg.bohr).magnitude

        grid_esp_0 = np.array(json.loads(
            conf_record.get_value(OEField("grid_esp_0_au_field", Types.String))
            )).reshape(-1)

        npoints = len(grid_bohr)

        r_ij = -(
            grid_bohr - geometry_bohr[:, None]
        )  # distance vector of grid_bohr points from atoms
        r_ij0 = cdist(grid_bohr, geometry_bohr, metric="euclidean")
        r_ij1 = np.power(r_ij0, -1)  # euclidean distance of atoms and grid_bohrs ^ -1
        r_ij3 = np.power(r_ij0, -3)  # euclidean distance of atoms and grid_bohrs ^ -3

        r_jk = (
            geometry_bohr - geometry_bohr[:, None]
        )  # distance vector of atoms from each other
        r_jk1 = cdist(
            geometry_bohr, geometry_bohr, metric="euclidean"
        )  # euclidean distance of atoms from each other
        r_jk3 = np.power(
            r_jk1, -3, where=r_jk1 != 0
        )  # euclidean distance of atoms from each other ^ -3

        oechem.OEPerceiveSymmetry(oemol, includeH=True)
        chemically_equivalent_atoms = [a.GetSymmetryClass() for a in oemol.GetAtoms()]
        chemically_equivalent_atoms_pairs = pair_equivalent(chemically_equivalent_atoms)

        n_chemically_equivalent_atoms = len(chemically_equivalent_atoms_pairs)
        net_charge = sum([a.GetFormalCharge() for a in oemol.GetAtoms()])

        coulomb14scale_matrix = coulomb_scaling_oe(oemol, coulomb14scale=0.5)

        forced_symmetry = set(
            [item for sublist in chemically_equivalent_atoms_pairs for item in sublist]
        )

        polar_region = list(set(range(natom)) - forced_symmetry)
        n_polar_region = len(polar_region)
        elements = [oechem.OEGetAtomicSymbol(a.GetAtomicNum()) for a in oemol.GetAtoms()]

        # Distance dependent matrices for fitting

        drjk = np.zeros((natom, natom, 3))
        for k in range(natom):
            drjk[k] = r_jk[k] * (r_jk3[k] * coulomb14scale_matrix[k]).reshape(-1, 1)

        # start charge-fitting
        # first stage, no symmetry
        ndim1 = natom + 1
        a0 = np.einsum("ij,ik->jk", r_ij1, r_ij1)
        a1 = np.zeros((ndim1, ndim1))
        a1[:natom, :natom] = a0

        # Lagrange multiplier
        a1[natom, :] = 1.0
        a1[:, natom] = 1.0
        a1[natom, natom] = 0.0

        b1 = np.zeros(ndim1)
        b1[:natom] = np.einsum("ik,i->k", r_ij1, grid_esp_0)
        b1[natom] = net_charge

        q1 = np.linalg.solve(a1, b1)[:natom]

        q11 = np.zeros(natom)

        while not np.allclose(q1, q11):
            a10 = copy.deepcopy(a1)
            for j in range(natom):
                if elements[j] != "H":
                    a10[j, j] += 0.0005 * np.power((q1[j] ** 2 + 0.1**2), -0.5)
            q1 = q11
            q11 = np.linalg.solve(a10, b1)[:natom]

        resp1 = q11

        # second stage, apply forced symmetry
        ndim2 = natom + 1 + n_chemically_equivalent_atoms + n_polar_region
        a2 = np.zeros((ndim2, ndim2))
        a2[: natom + 1, : natom + 1] = a1

        if n_chemically_equivalent_atoms == 0:
            pass
        else:
            for idx, pair in enumerate(chemically_equivalent_atoms_pairs):
                a2[natom + 1 + idx, pair[0]] = 1.0
                a2[natom + 1 + idx, pair[1]] = -1.0
                a2[pair[0], natom + 1 + idx] = 1.0
                a2[pair[1], natom + 1 + idx] = -1.0

        b2 = np.zeros(ndim2)
        b2[natom] = net_charge

        charge_to_be_fixed = q1[polar_region]

        for idx, pol_idx in enumerate(polar_region):
            a2[ndim2 - n_polar_region + idx, pol_idx] = 1.0
            a2[pol_idx, ndim2 - n_polar_region + idx] = 1.0
            b2[ndim2 - n_polar_region + idx] = charge_to_be_fixed[idx]

        q2 = resp1
        q22 = np.zeros(natom)
        steps = 0
        while not np.allclose(q2, q22) and steps < 20:
            a20 = copy.deepcopy(a2)
            for j in range(natom):
                if elements[j] != "H":
                    a20[j, j] += 0.001 * np.power((q2[j] ** 2 + 0.1**2), -0.5)

            desp = calc_desp(
                natom=natom, qs=q2, alphas=alphas, drjk=drjk, r_ij=r_ij, r_ij3=r_ij3
            )
            esp_to_fit = grid_esp_0 - desp
            b2[:natom] = np.einsum("ik,i->k", r_ij1, esp_to_fit)
            q2 = q22
            q22 = np.linalg.solve(a20, b2)[:natom]
            steps += 1

        resp2 = q22

        # quality of fit
        base_esp = np.dot(r_ij1, resp2)
        dpol_esp = calc_desp(
            natom=natom, qs=q2, alphas=alphas, drjk=drjk, r_ij=r_ij, r_ij3=r_ij3
        )
        final_esp = base_esp + dpol_esp

        ## rrms
        y = lambda x: np.sqrt(
            (sum((x - grid_esp_0) ** 2) / sum(grid_esp_0**2)) / npoints
        )

        dpol_rrms = y(final_esp)

        return resp1, resp2, dpol_rrms

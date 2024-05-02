#!/usr/bin/env python

import copy
import json
import os
from collections import defaultdict

import numpy as np

from dpolfit.utilities.constants import a03_to_angstrom3, a03_to_nm3

# https://github.com/numpy/numpy/issues/20895
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))

from datetime import datetime

import pint
from openeye import oechem
from openeye.oechem import OEField, Types
from scipy.optimize import nnls
from scipy.spatial.distance import cdist
from tqdm import tqdm

from dpolfit.utilities.miscellaneous import *

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def label_alpha(oemol: oechem.OEMol, smarts_pattern: str, index: bool = True) -> list:
    """
    Lable polarizability parameters on the molecule

    :param oemol: openeye molecule object
    :type oemol: OEMol
    :param smarts_pattern: the smarts pattern used to match the molecule
    :type smarts_pattern: string
    :return: return matched SMARTs patterns
    :rtype: list
    """
    ss = oechem.OESubSearch(smarts_pattern)
    oechem.OEPrepareSearch(oemol, ss)

    ret = []
    for match in ss.Match(oemol):
        # Only need the first atom, not the entire pattern
        if index:
            ret.append([ma.target.GetIdx() for ma in match.GetAtoms()][0])
        else:
            ret.append([ma.target for ma in match.GetAtoms()][0])

    return ret


def train(oedatabase: oechem.OEMolRecord, parameter_types: list) -> dict:
    """
    The main function to train polarizability against generated QM ESPs

    :param oedatabase: The OE database object that contains all training data
    :type oedatabase: .oedb
    :param parameter_types: The polarizability typing scheme
    :type parameter_types: List
    :return: return derived polarizabilities and optimization details
    :rtype: dict
    """
    ndim = len(parameter_types)
    positions = {parm: idx for idx, parm in enumerate(parameter_types)}
    param_count = {parm: 0 for parm in parameter_types}
    parameter_copy = copy.copy(parameter_types)

    A = np.zeros((ndim, ndim))
    B = np.zeros(ndim)

    ifs = oechem.oeifstream(oedatabase)

    dataset = []
    for oerecord in tqdm(list(oechem.read_mol_records(ifs)), desc="Fitting progress"):
        oemol = oerecord.get_mol()
        oesmi = oechem.OEMolToSmiles(oemol)
        dataset.append(oesmi)

        param_dict = defaultdict(list)
        included = []
        for idx, parm in enumerate(parameter_types):
            labelled = label_alpha(oemol=oemol, smarts_pattern=parm)

            if len(labelled) > 0:
                for atom in labelled:
                    if atom in included:
                        pass
                    else:
                        param_dict[parm].append(atom)
                        param_count[parm] += 1
                        included.append(atom)

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

            grid_esp_0 = json.loads(
                conf_record.get_value(OEField("grid_esp_0_au_field", Types.String))
            )

            r_ij = -(grid_bohr - geometry_bohr[:, None])
            r_ij3 = np.power(cdist(geometry_bohr, grid_bohr, metric="euclidean"), -3)

            r_jk = geometry_bohr - geometry_bohr[:, None]
            r_jk1 = cdist(geometry_bohr, geometry_bohr, metric="euclidean")
            r_jk3 = np.power(r_jk1, -3, where=r_jk1 != 0)

            this_A = np.zeros((ndim, ndim))

            for d, ef in perturb_dipoles.items():
                grid_esp_i = json.loads(
                    conf_record.get_value(
                        OEField(f"grid_esp_{d}_au_field", Types.String)
                    )
                )

                vdiff = np.array(grid_esp_i) - np.array(grid_esp_0)

                D_ijE = np.dot(r_ij, ef) * r_ij3

                for k, (smirks_k, alphas_k) in enumerate(param_dict.items()):
                    Cik = D_ijE[alphas_k].sum(axis=0)
                    B[positions[smirks_k]] += np.dot(Cik, vdiff).item()

                    for l, (smirks_l, alphas_l) in enumerate(param_dict.items()):
                        if k > l:
                            continue
                        Cil = D_ijE[alphas_l].sum(axis=0)
                        this_A[positions[smirks_k], positions[smirks_l]] += np.dot(
                            Cil, Cik
                        )

            this_A += this_A.T - np.diag(this_A.diagonal())

            A += this_A

    if np.where(B == 0)[0].size > 0:
        not_fitted = np.where(B == 0)[0]

        for i in sorted(not_fitted, reverse=True):
            nft = parameter_types.pop(i)

        fitted_positions = [positions[p] for p in parameter_types]

        print(
            f"{len(not_fitted)} not fitted parameters:",
            *[parameter_copy[p] for p in not_fitted],
            sep="\n",
        )

        A = A[tuple(np.meshgrid(fitted_positions, fitted_positions))]
        B = B[fitted_positions]

    else:
        pass

    # ret, _ = nnls(A, B)
    ret = np.linalg.solve(A, B)

    #dt_json = {"parameters": {p: v for p, v in zip(parameter_types, ret)}} | {
    dt_json = {"parameters": {p: v*a03_to_nm3 for p, v in zip(parameter_types, ret)}} | {
        #"_unit": "bohr**3",
         "_unit": "nm**3",
        "_generated": datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
        "_dataset": dataset,
        "design_matrix_count": param_count,
    }

    return dt_json

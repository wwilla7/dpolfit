#!/usr/bin/env python

import json
import os
import copy
from collections import defaultdict
import numpy as np

# https://github.com/numpy/numpy/issues/20895
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))

import pint
from openeye import oechem
from oechem import OEField, Types
from scipy.spatial.distance import cdist
from tqdm import tqdm
from datetime import datetime
from scipy.optimize import nnls

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

perturb_dipoles = {
    "x+": [0.01, 0.0, 0.0],
    "x-": [-0.01, 0.0, 0.0],
    "y+": [0.0, 0.01, 0.0],
    "y-": [0.0, -0.01, 0.0],
    "z+": [0.0, 0.0, 0.01],
    "z-": [0.0, 0.0, -0.01],
}

a03_to_angstrom3 = Q_(1, "a0**3").to("angstrom**3").magnitude
a03_to_nm3 = Q_(1, "a0**3").to("nm**3").magnitude


def label_alpha(oemol, smarts_pattern):
    ss = oechem.OESubSearch(smarts_pattern)
    oechem.OEPrepareSearch(oemol, ss)

    ret = []
    for match in ss.Match(oemol):
        # Only need the first atom, not the entire pattern
        ret.append([ma.target.GetIdx() for ma in match.GetAtoms()][0])

    return ret


def train(oedatabase, parameter_types):
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

    dt_json = {"parameters": {p: v for p, v in zip(parameter_types, ret)}} | {
        # dt_json = {"parameters": {p: v*a03_to_nm3 for p, v in zip(parameter_types, ret)}} | {
        "_unit": "bohr**3",
        # "_unit": "nm**3",
        "_generated": datetime.now().strftime("%m-%d-%Y %H:%M:%S"),
        "_dataset": dataset,
        "design_matrix_count": param_count,
    }

    return dt_json


if __name__ == "__main__":
    element_typed = [
        "[#1:1]",
        "[#6:1]",
        "[#7:1]",
        "[#8:1]",
        "[#9:1]",
        "[#15:1]",
        "[#16:1]",
        "[#17:1]",
        "[#35:1]",
    ]

    sagevdw_typed = [
        "[#1:1]-[#8X2H2+0]-[#1]",
        "[#1]-[#8X2H2+0:1]-[#1]",
        "[#53X0-1:1]",
        "[#35X0-1:1]",
        "[#17X0-1:1]",
        "[#9X0-1:1]",
        "[#55+1:1]",
        "[#37+1:1]",
        "[#19+1:1]",
        "[#11+1:1]",
        "[#3+1:1]",
        "[#53:1]",
        "[#35:1]",
        "[#17:1]",
        "[#9:1]",
        "[#15:1]",
        "[#16:1]",
        "[#7:1]",
        "[#8X2H1+0:1]",
        "[#8X2H0+0:1]",
        "[#8:1]",
        "[#6X4:1]",
        "[#6X2:1]",
        "[#6:1]",
        "[#1:1]-[#16]",
        "[#1:1]-[#8]",
        "[#1:1]-[#7]",
        "[#1:1]-[#6X2]",
        "[#1:1]-[#6X3](~[#7,#8,#9,#16,#17,#35])~[#7,#8,#9,#16,#17,#35]",
        "[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]",
        "[#1:1]-[#6X3]",
        "[#1:1]-[#6X4]~[*+1,*+2]",
        "[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])(-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]",
        "[#1:1]-[#6X4](-[#7,#8,#9,#16,#17,#35])-[#7,#8,#9,#16,#17,#35]",
        "[#1:1]-[#6X4]-[#7,#8,#9,#16,#17,#35]",
        "[#1:1]-[#6X4]",
        "[#1:1]",
    ]

    import importlib_resources

    data_path = importlib_resources.files("dpolfit").joinpath(
        os.path.join("data", "tests")
    )
    os.chdir(data_path)

    oedata_file = "dataset.oedb"

    ret = train(oedatabase=oedata_file, parameter_types=element_typed)
    ret |= {"_dataset_file": oedata_file}
    json.dump(ret, open("results.json", "w"), indent=2)

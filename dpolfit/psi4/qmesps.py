#!/usr/bin/env python

import json
import os
import shutil
from glob import glob

import numpy as np
import psi4
import ray
from numpyencoder import NumpyEncoder
from openeye import oechem, oeomega
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.grids import GridGenerator, MSKGridSettings
from openff.toolkit.topology import Molecule
from openff.units import unit

from dpolfit.utilities.miscellaneous import *


@ray.remote(num_cpus=8)
def psi4_optimizer(wd: str) -> float:
    """
    Optimize geometry before QM ESPs calcs

    :param wd: working directory
    :type wd: str
    :return: returned QM energy
    :rtype: float
    """
    os.chdir(wd)

    with open("input.xyz", "r") as f:
        mol_xyz = f.read()

    psi4.core.be_quiet()
    psi4.set_num_threads(8)
    psi4mol = psi4.geometry(mol_xyz)

    # psi4.set_options({"basis": "aug-cc-pvtz", "scf_type": "df"})
    # ene = psi4.optimize("mp2", molecule=psi4mol)

    # Reference for the choice of QM level of theory
    # DOI: 10.1021/acs.jcim.9b00962
    psi4.set_options({"basis": "cc-pV(T+d)Z", "scf_type": "df"})
    ene = psi4.optimize("b3lyp", molecule=psi4mol)

    psi4mol.save_xyz_file("optimized.xyz", 1)

    return ene


def generate_grid(wd: str, density: float = 17.0, layers: float = 10.0) -> 0:
    """
    [TODO:description]

    :param wd: [TODO:description]
    :type wd: str
    :return: [TODO:description]
    :rtype: 0
    """

    os.chdir(wd)

    grid_settings = MSKGridSettings(type="msk", density=density, layers=layers)

    # if os.path.exists("optimized.xyz"):
    #    pass
    # else:
    #    ret = psi4_optimizer.remote(wd)
    #    print(ray.get(ret))

    offmol = Molecule.from_file("input.sdf", file_format="sdf")
    conformer = (
        np.loadtxt("optimized.xyz", skiprows=2, usecols=(1, 2, 3)) * unit.angstrom
    )
    offmol.conformers.clear()
    offmol.conformers.append(conformer)

    grid = GridGenerator.generate(offmol, conformer, grid_settings)

    offmol.to_file("optimized.sdf", file_format="sdf")

    np.save("grid.npy", grid.magnitude)
    np.save("coordinates.npy", conformer.magnitude)

    return 0


@ray.remote(num_cpus=8)
def psi4_esps(
    imposed: str,
    wd: str,
    method: str = "mp2",
    basis: str = "aug-cc-pvtz",
    density: float = 17.0,
    layers: float = 10.0,
) -> 0:
    """
    Compute QM ESPs with Psi4

    :param imposed: imposed external electric field
    :type imposed: str
    :param wd: working directory
    :type wd: str
    :return: 0
    :rtype: float
    """
    cwd = os.getcwd()
    os.chdir(wd)

    qc_data_settings = ESPSettings(
        method=method,
        basis=basis,
        grid_settings=MSKGridSettings(type="msk", density=density, layers=layers),
        perturb_dipole=perturb_dipoles[imposed],
    )

    offmol = Molecule.from_file("optimized.sdf", file_format="sdf")

    grid = np.load("grid.npy") * unit.angstrom

    tmp_wd = os.path.join(os.getcwd(), imposed)
    if os.path.exists(tmp_wd):
        shutil.rmtree(tmp_wd)

    os.makedirs(tmp_wd, exist_ok=False)

    ret_conformer, esp, electric_field = Psi4ESPGenerator._generate(
        molecule=offmol,
        conformer=offmol.conformers[0],
        grid=grid,
        settings=qc_data_settings,
        directory=tmp_wd,
        minimize=False,
        compute_esp=True,
        compute_field=False,
        n_threads=8,
    )

    np.save(f"grid_esp.{imposed}.npy", esp.magnitude)

    os.chdir(cwd)

    return 0


def worker(input_file: str, wd: str, maxconf:int=10, imposed_fields:dict=perturb_dipoles) -> str:
    """
    The main function to carry out geometry optimization
    and QM ESPs generation.

    This function requires a modified version of `openff-recharge`
    for customized grid setting and imposed external electric fields.

    :param input_file: input dataset
    :type input_file: str
    :param wd: working directory
    :type wd: str
    :return: working directory that contains all the output data
    :rtype: str
    """

    cwd = os.getcwd()
    ifs = oechem.oemolistream(input_file)
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(maxconf)
    omega = oeomega.OEOmega(omegaOpts)
    workers = []
    for mol_idx, mol in enumerate(ifs.GetOEMols()):

        ret_code = omega.Build(mol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            m_p = os.path.join(wd, f"molecule{mol_idx:02d}")
            for idx, conf in enumerate(mol.GetConfs()):
                this_mol = oechem.OEMol(conf)
                c_p = os.path.join(m_p, f"conf{idx}")

                if os.path.exists(c_p):
                    shutil.rmtree(c_p)
                os.makedirs(c_p)
                shutil.copy2(input_file, c_p)
                os.chdir(c_p)
                ofs = oechem.oemolostream("input.xyz")
                oechem.OEWriteMolecule(ofs, this_mol)
                ofs = oechem.oemolostream("input.sdf")
                oechem.OEWriteMolecule(ofs, this_mol)
                ret = psi4_optimizer.remote(c_p)
                print(ray.get(ret))
                generate_grid(c_p)
                for k, v in imposed_fields.items():
                    ret = psi4_esps.remote(k, c_p)
                    workers.append(ret)
                os.chdir(cwd)

            # Save primary mol with no conformers
            oemolrecord = oechem.OEMolRecord()
            [a.SetMapIdx(a.GetIdx()) for a in mol.GetAtoms()]
            mapped_smiles = oechem.OECreateSmiString(
                mol, oechem.OESMILESFlag_AtomMaps | oechem.OESMILESFlag_Hydrogens
            )
            oemolrecord.set_value(mapped_smile_field, mapped_smiles)

            oemolrecord.set_mol(mol)
            rofs = oechem.oeofstream(os.path.join(m_p, "molecule.oedb"))
            oechem.OEWriteRecord(rofs, oemolrecord)

        else:
            oechem.OEThrow.Warning(
                "%s: %s" % (mol.GetTitle(), oeomega.OEGetOmegaError(ret_code))
            )

    ray.get(workers)

    molecules = glob(os.path.join(wd, "molecule*"))
    rofs = oechem.oeofstream(os.path.join(wd, "dataset.oedb"))
    for mol in molecules:
        ifs = oechem.oeifstream(os.path.join(mol, "molecule.oedb"))
        confs = glob(os.path.join(mol, "conf*"))
        for oemolrecord in oechem.read_mol_records(ifs):
            mapped_smiles = oemolrecord.get_value(mapped_smile_field)
            oemol = oemolrecord.get_mol()
            oemol.DeleteConfs()
            for conf in confs:
                coords = np.load(os.path.join(conf, "coordinates.npy"))
                oeconf = oemol.NewConf(oechem.OEFloatArray(coords.flatten().tolist()))

                conf_record = oemolrecord.get_conf_record(oeconf)
                conf_record.set_value(mapped_smile_field, mapped_smiles)
                conf_record.set_value(
                    geometry_angstrom_field, json.dumps(coords, cls=NumpyEncoder)
                )

                grids = np.load(os.path.join(conf, "grid.npy"))

                conf_record.set_value(
                    grid_angstrom_field, json.dumps(grids, cls=NumpyEncoder)
                )
                for d, e in imposed_fields.items():
                    grid_espi = np.load(os.path.join(conf, f"grid_esp.{d}.npy"))
                    conf_record.set_value(
                        OEField(f"grid_esp_{d}_au_field", OEStringType),
                        json.dumps(grid_espi, cls=NumpyEncoder),
                    )

                oemolrecord.set_conf_record(oeconf, conf_record)
                oemolrecord.set_mol(oemol)

        oechem.OEWriteRecord(rofs, oemolrecord)
    return wd

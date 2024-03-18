import json
import os
import shutil
from glob import glob

import numpy as np
import ray
from dpolfit.utilities.miscellaneous import *
from numpyencoder import NumpyEncoder
from openeye import oechem, oeomega
from openff.recharge.esp import ESPSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.recharge.grids import MSKGridSettings
from openff.toolkit.topology import Molecule
from openff.units import unit

import psi4


@ray.remote(num_cpus=8)
def psi4_optimizer(wd: str):

    os.chdir(wd)

    with open("input.xyz", "r") as f:
        mol_xyz = f.read()

    psi4.core.be_quiet()
    psi4.set_num_threads(8)
    psi4mol = psi4.geometry(mol_xyz)

    psi4.set_options({"basis": "aug-cc-pvtz", "scf_type": "df"})

    ene = psi4.optimize("mp2", molecule=psi4mol)
    psi4mol.save_xyz_file("optimized.xyz", 1)

    return ene


@ray.remote(num_cpus=8)
def psi4_esps(imposed: str, wd: str):
    cwd = os.getcwd()
    os.chdir(wd)

    if os.path.exists("optimized.xyz"):
        pass
    else:
        ret = psi4_optimizer.remote(wd)
        print(ray.get(ret))

    qc_data_settings = ESPSettings(
        method="mp2",
        basis="aug-cc-pvtz",
        grid_settings=MSKGridSettings(type="msk", density=17.0, layers=10.0),
        perturb_dipole=perturb_dipoles[imposed],
    )

    offmol = Molecule.from_file("input.sdf", file_format="sdf")
    conformer = (
        np.loadtxt("optimized.xyz", skiprows=2, usecols=(1, 2, 3)) * unit.angstrom
    )
    offmol.conformers.clear()
    offmol.conformers.append(conformer)

    ret_conformer, grid, esp, electric_field = Psi4ESPGenerator.generate(
        molecule=offmol,
        conformer=conformer,
        settings=qc_data_settings,
        directory=os.path.join(os.getcwd(), imposed),
        minimize=False,
        compute_esp=True,
        compute_field=False,
        n_threads=8,
    )

    offmol.to_file("optimized.sdf", file_format="sdf")

    if imposed == "0":
        np.save("grid.npy", grid.magnitude)
        np.save("coordinates.npy", conformer.magnitude)

    np.save(f"grid_esp.{imposed}.npy", esp.magnitude)

    os.chdir(cwd)

    return 0


def worker(input_file: str, wd: str):

    cwd = os.getcwd()
    ifs = oechem.oemolistream(input_file)
    omegaOpts = oeomega.OEOmegaOptions()
    omegaOpts.SetMaxConfs(10)
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
                for k, v in perturb_dipoles.items():
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
                for d, e in perturb_dipoles.items():
                    grid_espi = np.load(os.path.join(conf, f"grid_esp.{d}.npy"))
                    conf_record.set_value(
                        OEField(f"grid_esp_{d}_au_field", OEStringType),
                        json.dumps(grid_espi, cls=NumpyEncoder),
                    )

                oemolrecord.set_conf_record(oeconf, conf_record)
                oemolrecord.set_mol(oemol)

        oechem.OEWriteRecord(rofs, oemolrecord)
    return 0


if __name__ == "__main__":

    import importlib_resources

    ray_path = "/tmp/ray_tmp"
    os.makedirs(ray_path, exist_ok=True)
    ray.init(_temp_dir=ray_path)

    data_path = importlib_resources.files("dpolfit").joinpath("data")

    worker(
        input_file=os.path.join(data_path, "input.smi"),
        wd=os.path.join(data_path, "tests"),
    )

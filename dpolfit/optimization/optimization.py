#!/usr/bin/env python

import json
import os
import ray
import shutil
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Generator
import mdtraj as md
import numpy as np
import pandas as pd
import pint
import copy
from lxml import etree
from numpyencoder import NumpyEncoder
from openmm import (
    AmoebaMultipoleForce,
    LangevinIntegrator,
    NonbondedForce,
    Platform,
    XmlSerializer,
    unit,
    Vec3
)
from openmm.app import PDBFile, Simulation
from scipy.optimize import minimize
from glob import glob
from dpolfit.utilities.miscellaneous import create_monomer
from dpolfit.openmm.md import run, InputData
from dpolfit.utilities.constants import kb, na, kb_u, ureg, Q_

# ureg = pint.UnitRegistry()
# Q_ = ureg.Quantity


parameter_names = {
    "./HarmonicBondForce/Bond[@class1='401'][@class2='402']": "length",
    "./HarmonicAngleForce/Angle[@class1='402'][@class2='401'][@class3='402']": "angle",
    "./NonbondedForce/Atom[@class='401'][@sigma]": "sigma",
    "./NonbondedForce/Atom[@class='401'][@epsilon]": "epsilon",
    "./NonbondedForce/Atom[@class='402'][@sigma]": "sigma",
    "./NonbondedForce/Atom[@class='402'][@epsilon]": "epsilon",
    "./AmoebaMultipoleForce/Multipole[@type='401']": "c0",
    "./AmoebaMultipoleForce/Multipole[@type='402']": "c0",
    "./AmoebaMultipoleForce/Polarize[@type='401'][@polarizability]": "polarizability",
    "./AmoebaMultipoleForce/Polarize[@type='402'][@polarizability]": "polarizability",
    "./AmoebaMultipoleForce/Polarize[@thole]": "thole",
}


def _get_input(ff_file: str, parm_json: str) -> dict:
    """
    [TODO:description]

    :param ff_file: [TODO:description]
    :type ff_file: str
    :param parameters: [TODO:description]
    :type parameters: dict
    :return: [TODO:description]
    :rtype: dict
    """

    parameters = json.load(open(parm_json, "r"))

    tree = etree.parse(ff_file)
    root = tree.getroot()

    for k, v in parameters.items():
        ret = root.findall(k)
        for r in ret:
            dt = r.get(parameter_names[k])
            v.update({k: float(dt) for k in ["initial", "value"]})

    json.dump(parameters, open("parameters.json", "w"), indent=2)
    return parameters


def update_ffxml(ff_file: str, parameters: dict) -> str:
    tree = etree.parse(ff_file)
    root = tree.getroot()

    for k, v in parameters.items():
        # findall instead of find in case there are reused parameters, such as thole
        ret = root.findall(k)
        for p in ret:
            p.set(parameter_names[k], str(v["value"]))

    q1 = root.find("./AmoebaMultipoleForce/Multipole[@type='401']")
    q2 = root.find("./AmoebaMultipoleForce/Multipole[@type='402']")
    q2.set("c0", str(-0.5 * float(q1.attrib["c0"])))

    d = root.find(".//DateGenerated")
    d.text = f'{datetime.now().strftime("%m-%d-%Y %H:%M:%S")} on {os.uname().nodename}'

    etree.indent(tree, "  ")
    ret = etree.tostring(tree, encoding="utf-8", pretty_print=True).decode("utf-8")

    return ret


def update_from_template(input_array: np.array, parameters: dict) -> dict:
    """
    Update the parameters with new data split out from the optimizer

    :param input_array: [TODO:description]
    :type input_array: np.array
    :param parameters: [TODO:description]
    :type parameters: dict
    :return: [TODO:description]
    :rtype: dict
    """

    data = input_array.tolist()

    for k, v in parameters.items():
        v["value"] = data.pop(0)

    return parameters


def dipole_moments(
    positions: np.array,
    residues: Generator,
    charges: np.array,
    total_induced: np.array,
    **kwargs,
):

    data = []
    for res in residues:
        atom_indices = [a.index for a in res.atoms()]
        permanent_dipoles = positions[atom_indices] * charges[atom_indices].reshape(
            -1, 1
        )
        induced_dipoles = total_induced[atom_indices]
        total = permanent_dipoles + induced_dipoles
        total_norm = np.linalg.norm(total.sum(axis=0))
    data.append(total_norm)

    average_dipoles = Q_(np.mean(data), "nm*e").to("debye").magnitude

    return average_dipoles  # debye


def calc_properties(**kwargs):
    e_l = Q_(kwargs["lE"], "kJ/mole")
    e_g = Q_(kwargs["gE"], "kJ/mole")
    v_l = Q_(kwargs["v"], "nm**3")
    n = kwargs["nmols"]
    t = Q_(float(kwargs["temperature"]), ureg.kelvin)
    l_p = kwargs["l_p"]
    g_p = kwargs["g_p"]
    if "MPID" in list(kwargs.keys()):
        mpid = True
    else:
        mpid = False

    with open(os.path.join(l_p, "system.xml"), "r") as f:
        system = XmlSerializer.deserialize(f.read())

    amoeba = False
    forces = {}
    if mpid:
        import mpidplugin
        for idx, force in enumerate(system.getForces()):
            force_name = force.__class__.__name__
            force.setForceGroup(idx)
            forces[force_name] = idx
            if mpidplugin.MPIDForce.isinstance(force):
                MultForce = mpidplugin.MPIDForce.cast(force)
                amoeba = True

    else:
        for idx, force in enumerate(system.getForces()):
            force_name = force.__class__.__name__
            force.setForceGroup(idx)
            forces[force_name] = idx
            if isinstance(force, AmoebaMultipoleForce):
                MultForce = force
                amoeba = True
            if isinstance(force, NonbondedForce):
                qs = np.array(
                    [
                        force.getParticleParameters(i)[0] / unit.elementary_charge
                        for i in range(system.getNumParticles())
                    ]
                )


    pdb_file = os.path.join(l_p, "output.pdb")
    pdb = PDBFile(pdb_file)
    integrator = LangevinIntegrator(
        t.magnitude * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds
    )
    myplatform = Platform.getPlatformByName("CUDA")
    myproperties = {"Precision": "mixed"}
    simulation = Simulation(pdb.topology, system, integrator, myplatform, myproperties)

    traj = md.load(
        os.path.join(l_p, "trajectory.dcd"),
        top=pdb_file,
    )
    n_frames = traj.n_frames

    energies = defaultdict(list)

    mus = []
    for f in range(n_frames):
        box = traj.openmm_boxes(frame=f)
        positions = traj.openmm_positions(frame=f)
        simulation.context.setPeriodicBoxVectors(*box)
        simulation.context.setPositions(positions)

        if amoeba:
            mus.append(
                Q_(
                    MultForce.getSystemMultipoleMoments(simulation.context)[1:4],
                    "debye",
                )
                .to("e*a0")
                .magnitude
            )

        else:
            mus.append(
                Q_(np.sum(positions * np.array(qs).reshape(-1, 1), axis=0), "e*nm")
                .to("e*a0")
                .magnitude
            )

        for force_name, groupid in forces.items():
            energy = (
                simulation.context.getState(
                    getEnergy=True, groups={groupid}
                ).getPotentialEnergy()
                / unit.kilojoules_per_mole
            )
            energies[force_name].append(energy)

    json.dump(energies, open(os.path.join(l_p, "energies.json"), "w"))

    mus = np.array(mus)
    ### fluctuation of dipoles
    avg_sqr_mus_au = np.mean(np.square(mus), axis=0)
    avg_mus_sqr_au = np.square(np.mean(mus, axis=0))

    variance = np.sum(avg_sqr_mus_au - avg_mus_sqr_au)

    eps0 = (
        Q_(8.854187812e-12, "F/m").to("e**2/a0/hartree").magnitude
    )  # this is unitless
    # hartree = e**2/a0
    # 1/4pi

    prefactor = 1 / (3.0 * eps0 * kb.to("J/kelvin") * t).to("hartree").magnitude

    prefactor00 = 1 / eps0

    average_volumes = v_l.mean(axis=0).to("a0**3").magnitude
    eps = prefactor * variance * (1 / average_volumes)

    if amoeba:
        try:
            del qs
        except UnboundLocalError:
            print("Possibly using AmoebaVdwForce")
        parameters = [
            MultForce.getMultipoleParameters(ni)
            for ni in range(system.getNumParticles())
        ]
        if mpid:
            alphas = np.array([np.mean(p[-1]) for p in parameters])
            qs = np.array([p[0] for p in parameters])
        else:
            alphas = np.array([p[-1].value_in_unit(unit.nanometer**3) for p in parameters])
            qs = np.array([p[0].value_in_unit(unit.elementary_charge) for p in parameters])
        sum_alphas = Q_(np.sum(alphas), "nm**3").to("a0**3").magnitude
        eps_infty = prefactor00 * (1 / average_volumes) * sum_alphas + 1

        ret = {
            "eps_infty": eps_infty,
            "epsilon": eps + eps_infty,
        }

    else:
        ret = {
            "eps_infty": 0.0,
            "epsilon": eps + 1,
        }

    # H = E + PV
    pv_l = (Q_(1, ureg.atmosphere) * v_l / n / (1 / na)).to("kJ/mol")
    h_l = e_l + pv_l
    # kbT = (Q_(1, ureg.boltzmann_constant) * t).to(ureg.kilojoule)
    kbT = kb_u * t

    hvap = e_g + kbT - h_l.mean(axis=0) / n

    alpha = (
        1e4
        * (np.mean(h_l * v_l) - h_l.mean(axis=0) * v_l.mean(axis=0))
        / v_l.mean(axis=0)
        / kb_u
        / t
        / t
    )

    kappa = 1e6 * (1 / kb / t) * ((v_l**2).mean() - (v_l.mean()) ** 2) / v_l.mean()

    # dipole moments
    # gas phase
    gas_pdb = PDBFile(os.path.join(g_p, "output.pdb"))
    gas_positions = gas_pdb.positions
    gas_residues = gas_pdb.topology.residues()
    # there is not induced dipoles on molecules that don't contain 1-4 and above connectivity
    gas_natoms = gas_pdb.topology.getNumAtoms()

    with open(os.path.join(g_p, "system.xml"), "r") as f:
        gas_system = XmlSerializer.deserialize(f.read())
    for idx, force in enumerate(gas_system.getForces()):
        if mpid:
            if mpidplugin.MPIDForce.isinstance(force):
                GasMultForce = mpidplugin.MPIDForce.cast(force)
                gas_qs = np.array(
                    [
                        GasMultForce.getMultipoleParameters(ni)[0]
                        for ni in range(gas_natoms)
                    ]
                )
                gas_alphas = np.array(
                    [
                        np.mean(GasMultForce.getMultipoleParameters(ni)[-1])
                        for ni in range(gas_natoms)
                    ]
                )

        else:
            if isinstance(force, AmoebaMultipoleForce):
                GasMultForce = force
                gas_qs = np.array(
                    [
                        GasMultForce.getMultipoleParameters(ni)[0].value_in_unit(
                            unit.elementary_charge
                        )
                        for ni in range(gas_natoms)
                    ]
                )
                gas_alphas = np.array(
                    [
                        GasMultForce.getMultipoleParameters(ni)[-1].value_in_unit(
                            unit.nanometer**3
                        )
                        for ni in range(gas_natoms)
                    ]
                )
            if isinstance(force, NonbondedForce) and not amoeba:
                gas_qs = np.array(
                    [
                        force.getParticleParameters(i)[0] / unit.elementary_charge
                        for i in range(gas_natoms)
                    ]
                )
                gas_alphas = np.zeros(gas_natoms)

    if gas_natoms < 4 or mpid or not amoeba:
        induced = np.zeros((gas_natoms, 3), dtype=float)
    else:
        gas_integrator = LangevinIntegrator(
            t.magnitude * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds
        )
        gas_simulation = Simulation(
            gas_pdb.topology, gas_system, gas_integrator, myplatform, myproperties
        )
        gas_simulation.context.setPositions(gas_positions)
        induced = GasMultForce.getInducedDipoles(gas_simulation.context)


    gas_dipole = dipole_moments(
        positions=np.array(gas_positions.value_in_unit(unit.nanometer)),
        residues=gas_residues,
        charges=gas_qs,
        total_induced=np.array(induced),
    )

    # condensed phase

    condensed_positions = traj.openmm_positions(frame=n_frames - 1)
    condensed_residues = pdb.topology.residues()
    simulation.context.setPositions(condensed_positions)
    if amoeba and not mpid:
        induced = MultForce.getInducedDipoles(simulation.context)
    else:
        induced = np.zeros((system.getNumParticles(), 3), dtype=float)

    condensed_dipole = dipole_moments(
        positions=np.array(condensed_positions.value_in_unit(unit.nanometer)),
        residues=condensed_residues,
        charges=qs,
        total_induced=np.array(induced),
    )

    molpol = np.sum(gas_alphas) * 1000  # A^3

    ret |= {
        "hvap": hvap.magnitude,
        "alpha": alpha.magnitude,
        "kappa": kappa.to("1/bar").magnitude,
        "rho": kwargs["rho"],
        "gas_mu": gas_dipole,
        "condensed_mu": condensed_dipole,
        "molpol": molpol,
        "speed (ns/day)": kwargs["speed"],
        "nmols": kwargs["nmols"],
        "temperature": t.magnitude,
    }

    return ret


class Worker:
    def __init__(self, work_path: str, template_path: str, ngpus: int = 4):
        self.iteration = 1
        self.work_path = work_path
        self.template_path = template_path

        self.parameter_template = json.load(
            open(os.path.join(self.template_path, "parameters.json"), "r")
        )
        self.penalty_priors = np.array(
            [v["prior"] for _, v in self.parameter_template.items()]
        ).astype(float)
        self.prior = np.array(
            [v["initial"] for _, v in self.parameter_template.items()]
        ).astype(float)

        self.references = pd.read_csv(
            os.path.join(self.template_path, "references.csv"), comment="#"
        )

        wts = [c for c in self.references.columns if "_wt" in c]
        self.references[wts] = self.references[wts] / self.references[wts].values.sum()

        self.references["weight"] = self.references[wts].sum(axis=1)
        self.targets = self.references.loc[self.references["weight"] != 0]
        self.temperatures = self.targets["temperature"]

        # the force field xml template
        self.ff_file = os.path.join(self.template_path, "forcefield.xml")
        self.cwd = os.getcwd()

        self.input_pdb = os.path.join(self.template_path, "input.pdb")

        if os.path.exists(os.path.join(self.template_path, "input.json")):
            tmp0 = json.load(open(os.path.join(self.template_path, "input.json"), "r"))
            tmp1 = {
                k: tmp0[k] for k in list(InputData().__annotations__) if k in list(tmp0)
            }
            self.input_data = InputData(**tmp1)

        else:
            self.input_data = InputData()

        if ngpus > 1:
            self.ray = True

            ray.init(_temp_dir="/tmp/ray_tmp", num_gpus=ngpus, num_cpus=2 * ngpus)

            run_remote = ray.remote(num_gpus=1, num_cpus=2)(run)
            calc_properties_remote = ray.remote(num_cpus=2, num_gpus=1)(calc_properties)
            self.calcs = {
                True: {
                    "run": run_remote.remote,
                    "calc_properties": calc_properties_remote.remote,
                },
                False: {"run": run, "calc_properties": calc_properties},
            }

        else:
            self.ray = False
            self.calcs = {False: {"run": run, "calc_properties": calc_properties}}

    @staticmethod
    def _prepare_input(iter_path, temperature):

        l_p = os.path.join(iter_path, "l")
        g_p = os.path.join(iter_path, "g")
        l_log = pd.read_csv(os.path.join(l_p, "simulation.log"), skiprows=[1])
        g_log = pd.read_csv(os.path.join(g_p, "simulation.log"), skiprows=[1])

        l_pdb = PDBFile(os.path.join(l_p, "output.pdb"))
        nmol = l_pdb.topology.getNumResidues()

        ret = {
            "temperature": temperature,
            "lE": l_log["Potential Energy (kJ/mole)"].values,
            "gE": g_log["Potential Energy (kJ/mole)"].mean(axis=0),
            "rho": l_log["Density (g/mL)"].mean(axis=0),
            "speed": l_log["Speed (ns/day)"].mean(axis=0),
            "v": l_log["Box Volume (nm^3)"].values,
            "nmols": nmol,
            "l_p": l_p,
            "g_p": g_p,
        }
        return ret

    def worker(self, input_array, single=False, **kwargs):

        print("Running iteration:    ", self.iteration)

        # prepare new force field file with new parameters from the optimizer
        if single:
            with open(self.ff_file, "r") as f:
                new_ff = f.read()
                new_param = dict()
                self.prior = input_array
                self.penalty_priors = np.full_like(self.prior, 1)
        else:
            new_param = update_from_template(input_array, self.parameter_template)
            new_ff = update_ffxml(self.ff_file, new_param)

        # prepare simulation files and change to the directory
        iter_path = os.path.join(self.work_path, f"iter_{self.iteration:03d}")
        if os.path.exists(os.path.join(iter_path, "properties.csv")):
            dataframe = pd.read_csv(os.path.join(iter_path, "properties.csv"))
            objt = self.objective(calc_data=dataframe, current_params=input_array)
            print(f"Restarting, previously estimated objective: {objt:.5f}")
        else:

            os.makedirs(iter_path, exist_ok=True)
            os.chdir(iter_path)

            # dump current parameters and force field
            json.dump(new_param, open("parameters.json", "w"), indent=2)
            with open("forcefield.xml", "w") as f:
                f.write(new_ff)

            workers = []
            for temp in self.temperatures:
                temp_path = os.path.join(iter_path, str(np.floor(temp).astype(int)))
                for f, e, m in zip(["l", "g"], ["npt", "nvt"], [False, True]):
                    work_path = os.path.join(temp_path, f)
                    input_data = copy.deepcopy(self.input_data)
                    input_data.temperature = temp
                    input_data.ensemble = e
                    input_data.work_dir = work_path
                    os.makedirs(work_path, exist_ok=True)
                    shutil.copy2("forcefield.xml", work_path)
                    if m:
                        create_monomer(
                            self.input_pdb, os.path.join(work_path, "input.pdb")
                        )
                    else:
                        shutil.copy2(
                            self.input_pdb, os.path.join(work_path, "input.pdb")
                        )

                    workers.append(self.calcs[self.ray]["run"](input_data))

            if self.ray:
                workers = ray.get(workers)

            if "Failed" in workers:
                print("Failed to run simulation", new_param)
                objt = 9999

            else:
                workers = []
                for temp in self.temperatures:
                    temp_path = os.path.join(iter_path, str(np.floor(temp).astype(int)))
                    input_data = self._prepare_input(temp_path, temp)

                    workers.append(
                        self.calcs[self.ray]["calc_properties"](**input_data)
                    )

                if self.ray:
                    ret = ray.get(workers)

                else:
                    ret = workers

                dataframe = pd.DataFrame(ret)

                dataframe.to_csv(os.path.join(iter_path, "properties.csv"), index=False)

                objt = self.objective(calc_data=dataframe, current_params=input_array)

        with open(os.path.join(self.work_path, "results.csv"), "a") as f:
            f.write("{:03d},{}\n".format(self.iteration, objt))

        self.iteration += 1

        os.chdir(self.cwd)

        return objt

    def objective(self, calc_data: pd.DataFrame, current_params, **kwargs):
        nparams = len(current_params)
        properties = [
            "epsilon",
            "rho",
            "hvap",
            "alpha",
            "kappa",
            "gas_mu",
            "condensed_mu",
            "molpol",
        ]

        # absolute error percent
        abp = lambda x, y, z: (abs(x - y) / abs(y)) * z

        objt = np.array(
            [
                abp(
                    calc_data.loc[calc_data["temperature"] == t, p].values[0],
                    self.targets.loc[self.targets["temperature"] == t, p].values[0],
                    self.targets.loc[
                        self.targets["temperature"] == t, f"{p}_wt"
                    ].values[0],
                )
                for t, p in zip(self.targets["temperature"], properties)
            ]
        ).sum()

        p = (
            lambda x, y: np.square(
                (current_params[x] - self.prior[x]) / self.penalty_priors[x]
            )
            * y
        )

        ret_penalties = np.sum([p(i, objt) for i in range(nparams)])

        objt += ret_penalties

        # if np.where(calc_data["rho"] == calc_data["rho"].max())[0][0] < 1:
        #     objt += 100 * objt
        #
        # rho_diff = calc_data["rho"].max() - calc_data["rho"].min()
        # if rho_diff > 0.05:
        #     objt += 100 * objt
        # else:
        #     objt -= 100 * objt

        return objt

    def evaluate(self):
        iterations = glob(os.path.join(self.work_path, "iter_*"))
        niter = len(iterations)
        if niter == 1:
            self.prior = np.zeros(3)
            self.penalty_priors = np.full_like(self.prior, 1)
        for i in range(1, niter + 1, 1):
            workers = []
            print(f"re-evaluate iteration {i} of {niter} ...")
            iter_path = os.path.join(self.work_path, f"iter_{i:03d}")
            for temp in self.temperatures:
                temp_path = os.path.join(iter_path, str(np.floor(temp).astype(int)))
                input_data = self._prepare_input(temp_path, temp)
                workers.append(self.calcs[self.ray]["calc_properties"](**input_data))

            if self.ray:
                ret = ray.get(workers)

            else:
                ret = workers

            dataframe = pd.DataFrame(ret)

            dataframe.to_csv(os.path.join(iter_path, "properties.csv"), index=False)

    def optimize(self, opt_method="Nelder-Mead", bounds=None):
        res = minimize(
            self.worker,
            self.prior,
            method=opt_method,
            bounds=bounds,
            options={"maxiter": 50},
        )

        return res


if __name__ == "__main__":
    cwd = os.getcwd()

    work_path = os.path.join(cwd, "simulations")
    template_path = os.path.join(cwd, "templates")

    if os.path.exists(work_path):
        shutil.rmtree(work_path)

    os.makedirs(work_path)

    wworker = Worker(work_path=work_path, template_path=template_path)
    results = wworker.optimize()

    print(results)

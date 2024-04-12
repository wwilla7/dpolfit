#!/usr/bin/env python

import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Generator
import mdtraj as md
import numpy as np
import pandas as pd
import pint
from lxml import etree
from numpyencoder import NumpyEncoder
from openmm import (
    AmoebaMultipoleForce,
    LangevinIntegrator,
    NonbondedForce,
    Platform,
    XmlSerializer,
    unit,
)
from openmm.app import PDBFile, Simulation
from scipy.optimize import minimize
from glob import glob

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

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
    e_l = Q_(kwargs["Total Energy (kJ/mole)"], "kJ/mole")
    e_g = Q_(kwargs["Total Energy (kJ/mole) (gas)"], "kJ/mole")
    v_l = Q_(kwargs["Box Volume (nm^3)"], "nm**3")
    n = kwargs["nmols"]
    t = Q_(float(kwargs["temperature"]), ureg.kelvin)
    l_p = kwargs["l_p"]
    g_p = kwargs["g_p"]
    kb = Q_(1, ureg.boltzmann_constant).to("kJ/kelvin")
    na = Q_(1, "N_A")
    kb_u = (kb / (1 / na).to("mole")).to("kJ/kelvin/mole")

    with open(os.path.join(l_p, "system.xml"), "r") as f:
        system = XmlSerializer.deserialize(f.read())

    amoeba = False
    forces = {}
    for idx, force in enumerate(system.getForces()):
        force_name = force.__class__.__name__
        force.setForceGroup(idx)
        forces[force_name] = idx
        if isinstance(force, AmoebaMultipoleForce):
            MultForce = force
            amoeba = True
        if isinstance(force, NonbondedForce):
            qs = np.array([
                force.getParticleParameters(i)[0] / unit.elementary_charge
                for i in range(system.getNumParticles())
            ])

    pdb_file = os.path.join(l_p, kwargs["output_pdb"])
    pdb = PDBFile(pdb_file)
    integrator = LangevinIntegrator(
        t.magnitude * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds
    )
    myplatform = Platform.getPlatformByName("CUDA")
    myproperties = {"Precision": "mixed"}
    simulation = Simulation(pdb.topology, system, integrator, myplatform, myproperties)

    traj = md.load(
        os.path.join(
            l_p, f"{kwargs['trajectory']}_{kwargs['simulation_time_ns']}ns.dcd"
        ),
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
        del qs
        parameters = [
            MultForce.getMultipoleParameters(ni)
            for ni in range(system.getNumParticles())
        ]
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
        if isinstance(force, NonbondedForce) and not amoeba:
            gas_qs = np.array(
                [
                    force.getParticleParameters(i)[0] / unit.elementary_charge
                    for i in range(gas_natoms)
                ]
            )

    if gas_natoms < 4 or not amoeba:
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
        total_induced=induced,
    )

    # condensed phase

    condensed_positions = traj.openmm_positions(frame=n_frames - 1)
    condensed_residues = pdb.topology.residues()
    simulation.context.setPositions(condensed_positions)

    if amoeba:
        induced = MultForce.getInducedDipoles(simulation.context)
    else:
        induced = np.zeros((system.getNumParticles(), 3), dtype=float)

    condensed_dipole = dipole_moments(
        positions=np.array(condensed_positions.value_in_unit(unit.nanometer)),
        residues=condensed_residues,
        charges=qs,
        total_induced=np.array(induced),
    )

    ret |= {
        "hvap": hvap.magnitude,
        "alpha": alpha.magnitude,
        "kappa": kappa.to("1/bar").magnitude,
        "rho": kwargs["Density (g/mL)"],
        "gas_mu": gas_dipole,
        "condensed_mu": condensed_dipole,
        "Speed (ns/day)": kwargs["Speed (ns/day)"],
        "Box Volume (nm^3)": kwargs["Box Volume (nm^3)"].mean(axis=0),
    }

    return ret


class Worker:
    def __init__(self, work_path: str, template_path: str):
        self.iteration = 1
        self.work_path = work_path
        self.template_path = template_path

        self.parameter_template = json.load(
            open(os.path.join(self.template_path, "parameters.json"), "r")
        )
        self.penalty_priors = [v["prior"] for _, v in self.parameter_template.items()]
        self.prior = np.array([v["initial"] for _, v in self.parameter_template.items()])

        self.references = pd.read_csv(
            os.path.join(self.template_path, "references.csv")
        )
        self.properties = self.references["property"]
        self.experiment = self.references["expt"]
        _weights = self.references["weight"]
        self.weights = _weights / _weights.sum(axis=0)

        # the force field xml template
        self.ff_file = os.path.join(self.template_path, "forcefield.xml")
        self.cwd = os.getcwd()

    @staticmethod
    def _prepare_input(iter_path):

        input_data = json.load(open(os.path.join(iter_path, "l", "input.json"), "r"))
        l_p = os.path.join(iter_path, "l")
        g_p = os.path.join(iter_path, "g")
        l_log = pd.read_csv(os.path.join(l_p, "simulation.log"), skiprows=[1])
        g_log = pd.read_csv(os.path.join(g_p, "simulation.log"), skiprows=[1])

        l_pdb = PDBFile(os.path.join(l_p, "output.pdb"))
        nmol = l_pdb.topology.getNumResidues()

        input_data |= {
            "Total Energy (kJ/mole)": l_log["Total Energy (kJ/mole)"].values,
            "Total Energy (kJ/mole) (gas)": g_log["Total Energy (kJ/mole)"].mean(
                axis=0
            ),
            "Density (g/mL)": l_log["Density (g/mL)"].mean(axis=0),
            "Speed (ns/day)": l_log["Speed (ns/day)"].mean(axis=0),
            "Box Volume (nm^3)": l_log["Box Volume (nm^3)"].values,
            "nmols": nmol,
            "l_p": l_p,
            "g_p": g_p,
        }
        return input_data

    def worker(self, input_array, **kwargs):
        print("Running iteration:    ", self.iteration)

        # prepare new force field file with new parameters from the optimizer
        if kwargs["single"]:
            with open(self.ff_file, "r") as f:
                new_ff = f.read()
                new_param = dict()
        else:
            new_param = update_from_template(input_array, self.parameter_template)
            new_ff = update_ffxml(self.ff_file, new_param)

        # prepare simulation files and change to the directory
        iter_path = os.path.join(self.work_path, f"iter_{self.iteration:03d}")
        if os.path.exists(iter_path):
            ret = json.load(open(os.path.join(iter_path, "properties.json"), "r"))
            objt = ret["objective"]
            print(f"Restarting, previously estimated objective: {objt:.5f}")
        else:
            shutil.copytree(os.path.join(self.template_path, "run"), iter_path)
            os.chdir(iter_path)

            # dump current parameters and force field
            json.dump(new_param, open("parameters.json", "w"), indent=2)
            with open("forcefield.xml", "w") as f:
                f.write(new_ff)

            shutil.copy2("forcefield.xml", os.path.join(iter_path, "l"))
            shutil.copy2("forcefield.xml", os.path.join(iter_path, "g"))

            # run simulations
            os.system("sh runlocal.sh")

            input_data = Worker._prepare_input(iter_path)
            calced = calc_properties(**input_data)

            properties = {"properties": {p: calced[p] for p in self.properties}}
            properties |= {"current_params": input_array}

            objt = self.objective(**properties)

            properties |= {
                "objective": objt,
                "weights": {
                    p: self.weights[idx] for idx, p in enumerate(self.properties)
                },
                "expt": {
                    p: self.experiment[idx] for idx, p in enumerate(self.properties)
                },
                "temperature": input_data["temperature"],
            }
            json.dump(
                properties,
                open(os.path.join(iter_path, "properties.json"), "w"),
                indent=2,
                cls=NumpyEncoder,
            )

        self.iteration += 1

        os.chdir(self.cwd)

        return objt

    def objective(self, **kwargs):
        current_params = kwargs["current_params"]
        nparams = len(current_params)

        # absolute error percent
        abp = lambda x, y, z: (abs(x - y) / abs(y)) * z

        objt = np.array(
            [
                abp(a, b, c)
                for a, b, c in zip(
                    [kwargs["properties"][p] for p in self.properties],
                    self.experiment,
                    self.weights,
                )
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

        return objt

    def evaluate(self):
        iterations = glob(os.path.join(self.work_path, "iter_*"))
        niter = len(iterations)
        for i in range(1, niter, 1):
            print(f"re-evaluate iteration {i} of {niter} ...")
            iter_path = os.path.join(self.work_path, f"iter_{i:03d}")
            input_data = Worker._prepare_input(iter_path)
            prp = calc_properties(**input_data)
            properties = json.load(
                open(os.path.join(iter_path, "properties.json"), "r")
            )

            properties["properties"].update({p: prp[p] for p in self.properties})
            objt = self.objective(**properties)
            properties.update(
                {
                    "objective": objt,
                    "weights": {
                        p: self.weights[idx] for idx, p in enumerate(self.properties)
                    },
                    "expt": {
                        p: self.experiment[idx] for idx, p in enumerate(self.properties)
                    },
                }
            )

            json.dump(
                properties,
                open(os.path.join(iter_path, "properties.json"), "w"),
                indent=2,
                cls=NumpyEncoder,
            )

    def optimize(self, opt_method="Nelder-Mead", bounds=None):
        res = minimize(self.worker, self.prior, method=opt_method, bounds=bounds)

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

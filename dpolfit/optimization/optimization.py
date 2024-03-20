#!/usr/bin/env python

import json
import os
import shutil
from collections import defaultdict

import mdtraj as md
import numpy as np
import pandas as pd
import pint
import scipy
from lxml import etree
from openmm import (AmoebaMultipoleForce, LangevinIntegrator, NonbondedForce,
                    Platform, XmlSerializer, unit)
from openmm.app import PDBFile, Simulation
from scipy.optimize import minimize

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

from dpolfit.openmm.ffxml import update_ff, update_results


def calc_properties(**kwargs) -> dict:
    """
    Post process MD simulations and calculate liquid properties

    :return: Return calculated properties
    :rtype: dict
    """
    e_l = Q_(kwargs["Total Energy (kJ/mole)"], "kJ/mole")
    e_g = Q_(kwargs["Total Energy (kJ/mole) (gas)"], "kJ/mole")
    v_l = Q_(kwargs["Box Volume (nm^3)"], "nm**3")
    n = kwargs["nmols"]
    t = Q_(float(kwargs["temperature"]), ureg.kelvin)
    l_p = kwargs["l_p"]
    kb = Q_(1, ureg.boltzmann_constant).to("kJ/kelvin")
    na = Q_(1, "N_A")
    kb_u = (kb / (1 / na).to("mole")).to("kJ/kelvin/mole")
    ret_p = kwargs["target_p"]

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
            qs = [
                force.getParticleParameters(i)[0] / unit.elementary_charge
                for i in range(system.getNumParticles())
            ]

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
        parameters = [
            MultForce.getMultipoleParameters(ni)
            for ni in range(system.getNumParticles())
        ]
        alphas = np.array([p[-1] / (unit.nanometer**3) for p in parameters])
        sum_alphas = Q_(np.sum(alphas), "nm**3").to("a0**3").magnitude
        eps_infty = prefactor00 * (1 / average_volumes) * sum_alphas + 1

        ret = {
            "High Frequency Dielectric": eps_infty,
            "Dielectric Constant": eps + eps_infty,
        }

    else:
        ret = {
            "High Frequency Dielectric": 0.0,
            "Dielectric Constant": eps + 1,
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

    ret |= {
        "Hvap (kJ/mol)": hvap.magnitude,
        "Thermal Expansion (10^-4 K^-1)": alpha.magnitude,
        "Isothermal Compressibility (10^-6 bar^-1)": kappa.to("1/bar").magnitude,
        "Density (g/mL)": kwargs["Density (g/mL)"],
        "Speed (ns/day)": kwargs["Speed (ns/day)"],
        "Box Volume (nm^3)": kwargs["Box Volume (nm^3)"].mean(axis=0),
    }

    return ret


def objective(**kwargs) -> float:
    """
    Calculate the objecitve of this optimization

    :return: the objective
    :rtype: float
    """
    # https://pubs.acs.org/doi/10.1021/jz500737m
    eps = 78.5
    density = 0.997  # g/mL
    hvap = 44.01568  # (kJ/mol)
    alpha = 2.56
    kappa = 45.3

    calc_eps = kwargs["Dielectric Constant"]
    calc_density = kwargs["Density (g/mL)"]
    calc_hvap = kwargs["Hvap (kJ/mol)"]
    calc_alpha = kwargs["Thermal Expansion (10^-4 K^-1)"]
    calc_kappa = kwargs["Isothermal Compressibility (10^-6 bar^-1)"]

    # absolute error percent
    abp = (
        lambda x, y: abs(x - y) / y
    )  # W: E731 do not assign a lambda expression, use a def

    objt = np.array(
        [
            abp(a, b)
            for a, b in zip(
                [calc_eps, calc_density, calc_hvap, calc_alpha, calc_kappa],
                [eps, density, hvap, alpha, kappa],
            )
        ]
    ).mean()
    return objt


class Worker:
    def __init__(self, work_path: str):
        """
        Perform optimization

        :param work_path: the path to store all intermediate data
        :type work_path: str
        """
        self.iteration = 1
        self.work_path = work_path

    def worker(self, input_array: np.array, **kwargs) -> float:
        """
        The main function to perform optimization

        :param input_array: input array splitted out by scipy optimizer
        :type input_array: np.array
        :return: return the objective to feed to the optimizer
        :rtype: float
        """

        print("Running iteration:    ", self.iteration)

        param_template = json.load(open("parameters.json", "r"))

        new_param = update_results(input_array, param_template)

        new_ff = update_ff("forcefield.xml", new_param)

        cwd = os.getcwd()

        iter_path = os.path.join(self.work_path, f"iter_{self.iteration:03d}")

        shutil.copytree(os.path.join(cwd, "run_scripts"), iter_path)

        os.chdir(iter_path)

        json.dump(new_param, open("parameters.json", "w"))

        with open("forcefield.xml", "w") as f:
            f.write(new_ff)

        shutil.copy2("forcefield.xml", os.path.join(iter_path, "l"))
        shutil.copy2("forcefield.xml", os.path.join(iter_path, "g"))

        os.system("sh runlocal.sh")

        input_data = json.load(open(os.path.join(iter_path, "l", "input.json"), "r"))
        l_p = os.path.join(iter_path, "l")
        g_p = os.path.join(iter_path, "g")
        l_log = pd.read_csv(os.path.join(l_p, "simulation.log"), skiprows=[1])
        g_log = pd.read_csv(os.path.join(g_p, "simulation.log"), skiprows=[1])

        l_pdb = PDBFile(os.path.join(l_p, "output.pdb"))
        nmol = l_pdb.topology.getNumResidues()

        ff_path = os.path.join(l_p, input_data["forcefield"])

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
            "target_p": iter_path,
            "ff_p": ff_path,
        }

        properties = calc_properties(**input_data)

        objt = objective(**properties)

        properties |= {"objective": objt}

        json.dump(properties, open("properties.json", "w"), indent=2)

        self.iteration += 1

        os.chdir(cwd)

        return objt

    def optimize(self, prior: np.array) -> scipy.optimize.OptimizeResult:
        """
        Perform optimization with scipy minimizer "Nelder-Mead"

        :param prior: Guess parameters
        :type prior: np.array
        :return: return the optimization result object
        :rtype: scipy.optimize.OptimizeResult
        """
        res = minimize(
            self.worker,
            prior,
            method="Nelder-Mead",
            bounds=(
                (0.0957, 0.1),
                (1.82, 1.91),
                (0.315, 0.318),
                (0.59, 0.684),
                (-0.89517, -0.675),
            ),
        )
        """
        bounds:
        bond length
        angle
        sigma
        epsilon
        401 c0
        """
        return res


if __name__ == "__main__":

    cwd = os.getcwd()
    prior = np.loadtxt(os.path.join(cwd, "prior.dat"))

    work_path = os.path.join(cwd, "simulations")

    if os.path.exists(work_path):
        shutil.rmtree(work_path)

    os.makedirs(work_path)

    wworker = Worker(work_path=work_path)
    results = wworker.optimize(prior)

    print(resutls)

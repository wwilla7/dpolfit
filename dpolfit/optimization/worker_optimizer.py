#!/usr/bin/env python

import logging
import os
from collections import defaultdict
from sys import stdout
from typing import Generator, Dict, List, Literal, Tuple, Union

import numpy as np
import openmm.unit as omm_unit
import ray
from dpolfit.optimization.utils import (
    Ensemble,
    IterationRecord,
    Properties,
    SimulationOutput,
    SimulationSettings,
    compute,
    compute_DielectricProperties,
    compute_hvap_alpha_kappa,
    create_serialized_system,
    get_custom_parms,
    read_openmm_output,
    set_custom_parms,
)
import shutil
from openff.interchange import Interchange
from openff.toolkit import ForceField as OForceField
from openff.toolkit import Topology as OTopology
from openmm import Context, LangevinIntegrator, Platform, System, XmlSerializer
from openmm.app import PDBFile
from scipy.optimize import minimize
from dpolfit.openmm.md import run

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Worker:
    def __init__(self, work_path: str, ngpus: int = 1):
        self.interation: IterationRecord = IterationRecord(
            iteration_number=1, loss_function=999
        )
        self.work_path = work_path
        if ngpus > 1:
            self.ray = True
            ray.init(_temp_dir="/tmp/ray_tmp", num_gpus=ngpus, num_cpus=2 * ngpus)

            run_remote = ray.remote(num_gpus=1, num_cpus=2)(run)
            compute_remote = ray.remote(num_gpus=1, num_cpus=2)(compute)
            self.compute_methods = {
                True: {"run": run_remote.remote, "compute": compute_remote.remote},
                False: {"run": run, "compute": compute},
            }

        else:
            self.ray = False
            self.compute_methods = {False: {"run": run, "compute": compute}}

    @classmethod
    def objective(
        cls,
        calc_data: Properties,
        ref_data: Properties,
        new_params: np.ndarray,
        ini_params: np.ndarray,
        penalty_priors: Union[np.ndarray, None] = None,
    ):
        nparam = len(new_params)

        # Same tolerance as OPC3
        # tol = 0.005 for density
        # tol = 0.025 for others

        # absolute errors
        # offset 10
        abp = lambda x, y, z: abs(x - y) / (y * z) / 10

        objt = 0

        for prop, value in calc_data.__dict__.items():
            if not np.isnan(value):
                if prop == "Density":
                    z = 0.005
                else:
                    z = 0.025
                    x = value.magnitude
                    y = getattr(ref_data, prop).magnitude
                    this_objt = abp(x, y, z) if abp(x, y, z) < 1 else 1
                    objt += this_objt

        if isinstance(penalty_priors, np.ndarray):

            # harmonic penalty function per parameter
            # Gaussian prior distribution to prevent
            # big deviation from the original values

            p = (
                lambda x, y: np.square(
                    (new_params[x] - ini_params[x]) / penalty_priors[x]
                )
                * y
            )

            parameter_penalty = np.sum([p(i, objt) for i in range(nparam)])
            objt += parameter_penalty

        return objt

    def setup(
        self,
        simulation_settings: List[SimulationSettings],
        reference_data: Dict[Union[float, int], Properties],
        forcefield: OForceField,
        topology: Dict[str, OTopology],
        use_last_percent: int = 50,
    ):
        self.simulation_settings = simulation_settings
        self.reference_data = reference_data
        self.forcefield = forcefield
        self.to_custom, self.ini_params = get_custom_parms(self.forcefield)
        self.topology = topology
        self.use_last_percent = use_last_percent

    def worker(self, input_array, penalty_priors=None):
        logging.debug(input_array)
        this_iteration = self.interation.iteration_number
        logging.info(f"Running iteration {this_iteration}")
        iter_path = os.path.join(self.work_path, f"iter_{this_iteration:02d}")
        # TODO restart the optimization
        if os.path.exists(iter_path):
            shutil.rmtree(iter_path)
            logging.info(f"remove existing {iter_path}")
        os.makedirs(iter_path)
        os.chdir(iter_path)

        # run all simulation
        forcefield = set_custom_parms(self.forcefield, self.to_custom, input_array)
        forcefield.to_file("custom.offxml")
        interchange_gas = Interchange.from_smirnoff(
            force_field=forcefield, topology=self.topology["gas"]
        )

        interchange_condensed = Interchange.from_smirnoff(
            force_field=forcefield, topology=self.topology["condensed"]
        )

        simulations = [
            (
                create_serialized_system(interchange_gas, settings)
                if settings.ensemble == Ensemble.NVT
                else create_serialized_system(interchange_condensed, settings)
            )
            for settings in self.simulation_settings
        ]

        workers = [
            self.compute_methods[self.ray]["run"](*simulation, settings)
            for simulation, settings in zip(simulations, self.simulation_settings)
        ]

        if self.ray:
            workers = ray.get(workers)

        if "Failed" in workers:
            logging.info("Failed to run simulation with parameters")
            logging.info(input_array)
            objt = 999

        else:
            compute_workers = []
            temperatures = defaultdict(list)
            for settings in self.simulation_settings:
                temp = settings.temperature.to("kelvin").magnitude
                temperatures[temp].append(settings)

            for k, v in temperatures.items():
                for s in v:
                    if s.ensemble.name == Ensemble.NVT.name:
                        gas_mdLog = read_openmm_output(
                            openmm_output=os.path.join(
                                iter_path, s.work_path, "simulation.csv"
                            ),
                            use_last_percent=self.use_last_percent,
                        )
                    elif s.ensemble.name == Ensemble.NPT.name:
                        liquid_mdLog = read_openmm_output(
                            openmm_output=os.path.join(
                                iter_path, s.work_path, "simulation.csv"
                            ),
                            use_last_percent=self.use_last_percent,
                        )
                        system_file = os.path.join(iter_path, s.work_path, "system.xml")
                        topology_file = os.path.join(
                            iter_path, s.work_path, "output.pdb"
                        )
                        trajectory_file = os.path.join(
                            iter_path, s.work_path, "trajectory.dcd"
                        )
                        liquid_mdSettings = s

                compute_workers.append(
                    self.compute_methods[self.ray]["compute"](
                        gas_mdLog=gas_mdLog,
                        liquid_mdLog=liquid_mdLog,
                        liquid_mdSettings=liquid_mdSettings,
                        system_file=system_file,
                        topology_file=topology_file,
                        trajectory_file=trajectory_file,
                        use_last_percent=self.use_last_percent,
                        calc_hvap=True,
                        MPID=True,
                    )
                )

            if self.ray:
                results = ray.get(compute_workers)

            else:
                results = compute_workers
            objt = 0
            for result in results:
                temp = result.Temperature.magnitude
                this_objective = Worker.objective(
                    calc_data=result,
                    ref_data=self.reference_data[temp],
                    new_params=input_array,
                    ini_params=self.ini_params,
                    penalty_priors=penalty_priors,
                )
                objt += this_objective

        self.interation = IterationRecord(
            iteration_number=this_iteration + 1, loss_function=objt
        )
        logging.info(f"iteration: {this_iteration}, loss_function: {objt:.5f}")

        return objt

    def optimize(
        self,
        opt_method: str = "Nelder-Mead",
        bounds: Union[np.ndarray, None] = None,
        penalty_priors: Union[np.ndarray, None] = None,
    ):
        res = minimize(
            self.worker,
            x0=self.ini_params,
            args=(penalty_priors),
            method=opt_method,
            bounds=bounds,
            options={"maxiter": 25},
        )

        return res

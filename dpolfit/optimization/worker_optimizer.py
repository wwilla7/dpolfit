#!/usr/bin/env python

import logging
import os
import json
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
    ureg,
    Q_,
)
import shutil
from openff.interchange import Interchange
from openff.toolkit import ForceField as OForceField
from openff.toolkit import Topology as OTopology
from openmm import Context, LangevinIntegrator, Platform, System, XmlSerializer
from openmm.app import PDBFile
from scipy.optimize import minimize
from dpolfit.openmm.md import run
from dataclasses import dataclass, replace

logger = logging.getLogger(__name__)


@dataclass
class OptimizationSettings(SimulationSettings):
    molecule_smiles: str = "CO"
    system_id: str = "CO_300"


class Worker:
    def __init__(self, work_path: str, ngpus: int = 1):
        self.iteration: IterationRecord = IterationRecord(
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
        weights: Union[Dict, None] = None,
    ):
        nparam = len(new_params)

        abp = lambda x, y, z: z * abs((x - y) / y)
        # abp = lambda x, y, z: abs(x - y) / (y * z) / 10

        objt = 0
        evaluated_properties = [
            k
            for k, v in ref_data.__dict__.items()
            if not np.isnan(v) and k not in ["Temperature", "SimulationTime"]
        ]
        n_prop = len(evaluated_properties)

        default_weights = {prop: 1 / n_prop for prop in evaluated_properties}
        # TODO give more spefic weights for the last two conditions
        if weights is None:
            weights = default_weights
        elif len(weights.keys()) != n_prop:
            weights = default_weights
        elif np.sum(weights.values()) != 1:
            weights = default_weights

        logger.info(weights)
        logger.info("property objective:")
        for prop in evaluated_properties:
            value = getattr(calc_data, prop)
            if not np.isnan(value):

                x = value.magnitude
                y = getattr(ref_data, prop).magnitude
                z = weights[prop]
                this_objt = abp(x, y, z)
                objt += this_objt
                logger.info(prop)
                logger.info(this_objt)

        if isinstance(penalty_priors, np.ndarray):

            # harmonic penalty function per parameter
            # Gaussian prior distribution to prevent
            # big deviation from the original values

            # p = (
            #     lambda x, y: np.square(
            #         (new_params[x] - ini_params[x]) / penalty_priors[x]
            #     )
            #     * y
            # )
            p = lambda x: penalty_priors[x] * np.square(new_params[x] - ini_params[x])

            parameter_penalty = np.sum([p(i) for i in range(nparam)])
            logger.info("penalty objective")
            logger.info(parameter_penalty)
            objt += parameter_penalty

        return objt

    def setup(
        self,
        simulation_settings: List[OptimizationSettings],
        reference_data: Dict[str, Properties],
        forcefield: OForceField,
        topology: Dict[str, OTopology],
        use_last_percent: int = 50,
        n_block: int = 1,
        eq_time: float = 1.0,
        MPID: bool = True,
    ):
        self.simulation_settings = simulation_settings
        self.reference_data = reference_data
        self.forcefield = forcefield
        self.to_custom, self.ini_params = get_custom_parms(self.forcefield)
        self.topology = topology
        self.use_last_percent = use_last_percent
        self.n_block = n_block
        self.eq_time = eq_time
        self.mpid = MPID

    def worker(self, input_array, penalty_priors=None, weights=None):
        this_iteration = self.iteration.iteration_number
        logger.info(f"Running iteration {this_iteration}")
        logger.info(input_array)
        iter_path = os.path.join(self.work_path, f"iter_{this_iteration:02d}")
        simulation_settings = [
            replace(s, work_path=os.path.join(iter_path, s.work_path))
            for s in self.simulation_settings
        ]

        # TODO restart the optimization
        if os.path.exists(iter_path):
            run_md = False

        else:
            run_md = True
            os.makedirs(iter_path)

        os.chdir(iter_path)

        # run all simulation
        forcefield = set_custom_parms(self.forcefield, self.to_custom, input_array)
        forcefield.to_file("custom.offxml")
        interchange_gas = {
            k: Interchange.from_smirnoff(force_field=forcefield, topology=t)
            for k, t in self.topology["gas"].items()
        }

        interchange_condensed = {
            k: Interchange.from_smirnoff(force_field=forcefield, topology=t)
            for k, t in self.topology["condensed"].items()
        }

        simulations = [
            [
                (
                    create_serialized_system(
                        interchange_gas[settings.system_id], settings
                    )
                    if settings.ensemble == Ensemble.NVT
                    else create_serialized_system(
                        interchange_condensed[settings.system_id], settings
                    )
                ),
                settings,
            ]
            for settings in simulation_settings
        ]

        if run_md:
            workers = [
                self.compute_methods[self.ray]["run"](
                    *simulation, settings, eq_time=self.eq_time
                )
                for simulation, settings in simulations
            ]

            if self.ray:
                workers = ray.get(workers)

        else:
            logger.info(f"using existing MD run in {iter_path}")
            workers = [
                (
                    "Failed"
                    if os.path.getsize(os.path.join(setting.work_path, "output.pdb"))
                    == 0
                    else 0
                )
                for setting in simulation_settings
            ]
        if "Failed" in workers:
            logger.info("Failed to run simulation with parameters")
            logger.info(input_array)
            objt = 999

        else:
            compute_workers = {}
            training_systems = defaultdict(list)
            for settings in simulation_settings:
                this_mol = settings.system_id
                training_systems[this_mol].append(settings)

            for k, v in training_systems.items():
                for s in v:
                    if s.ensemble.name == Ensemble.NVT.name:
                        gas_mdLog = read_openmm_output(
                            openmm_output=os.path.join(s.work_path, "simulation.csv"),
                            use_last_percent=self.use_last_percent,
                        )
                    elif s.ensemble.name == Ensemble.NPT.name:
                        liquid_mdLog = read_openmm_output(
                            openmm_output=os.path.join(s.work_path, "simulation.csv"),
                            use_last_percent=self.use_last_percent,
                        )
                        system_file = os.path.join(s.work_path, "system.xml")
                        topology_file = os.path.join(s.work_path, "output.pdb")
                        trajectory_file = os.path.join(
                            iter_path, s.work_path, "trajectory.dcd"
                        )
                        liquid_mdSettings = s

                try:
                    gas_mdLog
                except NameError:
                    gas_mdLog = SimulationOutput(
                        potential_energy=Q_([0.0] * 10, "kcal/mol")
                    )

                compute_workers[k] = self.compute_methods[self.ray]["compute"](
                    gas_mdLog=gas_mdLog,
                    liquid_mdLog=liquid_mdLog,
                    liquid_mdSettings=liquid_mdSettings,
                    system_file=system_file,
                    topology_file=topology_file,
                    trajectory_file=trajectory_file,
                    use_last_percent=self.use_last_percent,
                    calc_hvap=True,
                    MPID=self.mpid,
                    n_block=self.n_block,
                )

            if self.ray:
                results = {k: ray.get(v) for k, v in compute_workers.items()}

            else:
                results = compute_workers
            objt = 0
            data = []
            for sid, (result, ci) in results.items():
                this_objective = Worker.objective(
                    calc_data=result,
                    ref_data=self.reference_data[sid],
                    new_params=input_array,
                    ini_params=self.ini_params,
                    penalty_priors=penalty_priors,
                    weights=weights,
                )
                objt += this_objective
                data.append(
                    {k: v.magnitude for k, v in result.__dict__.items()}
                    | {
                        "CI": ci.magnitude,
                        "objt": this_objective,
                        "system_id": sid,
                    }
                )
            with open("results.json", "w") as f:
                json.dump(data, f, indent=2)

        self.iteration = IterationRecord(
            iteration_number=this_iteration + 1, loss_function=objt
        )
        logger.info(f"iteration: {this_iteration}, loss_function: {objt:.5f}")

        return objt

    def optimize(
        self,
        opt_method: str = "Nelder-Mead",
        bounds: Union[np.ndarray, None] = None,
        penalty_priors: Union[np.ndarray, None] = None,
        weights: Union[Dict, None] = None,
    ):
        res = minimize(
            self.worker,
            x0=self.ini_params,
            args=(penalty_priors, weights),
            method=opt_method,
            bounds=bounds,
            # options={"maxiter": 50},
        )

        return res

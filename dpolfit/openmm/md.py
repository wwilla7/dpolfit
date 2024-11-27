import os

from pydantic.dataclasses import Field, dataclass

import openmm
from openmm import (
    AmoebaMultipoleForce,
    LangevinIntegrator,
    MonteCarloBarostat,
    Platform,
    XmlSerializer,
    unit,
)
from openmm.app import (
    PME,
    DCDReporter,
    ForceField,
    HBonds,
    NoCutoff,
    PDBFile,
    PDBReporter,
    Simulation,
    StateDataReporter,
)
from dpolfit.optimization.utils import SimulationSettings, Ensemble
try:
    import mpidplugin
except ImportError as e:
    print(e)
from sys import stdout
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@dataclass(config=dict(validate_assignment=True))
class InputData:
    input_pdb: str = "input.pdb"
    forcefield: str = "forcefield.xml"
    polarization: str = "direct"
    cuda_device: str = "0"
    simulation_time_ns: float = 1
    temperature: float = 298.15
    restart: str = "restart.xml"
    rigidWater: bool = True
    timestep: float = 2.0
    ensemble: str = "npt"
    work_dir: str = "None"


def _run(input_data: InputData):
    """
    Run a simple plain MD with input information

    :param input_data: input information
    :type input_data: InputData
    """

    if input_data.work_dir == "None":
        work_dir = os.getcwd()
    else:
        work_dir = input_data.work_dir
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)

    ff_file = input_data.forcefield.split(" ")

    forcefield = ForceField(*ff_file)
    timestep = input_data.timestep
    temperature = input_data.temperature * unit.kelvin
    deviceid = str(input_data.cuda_device)

    pdb = PDBFile(input_data.input_pdb)
    top = pdb.topology
    positions = pdb.positions

    if input_data.ensemble.lower() == "npt":
        npt = True
        system_arguments = {
            "topology": top,
            "nonbondedMethod": PME,
            "nonbondedCutoff": 0.9 * unit.nanometer,
            "constraints": HBonds,
            "rigidWater": input_data.rigidWater,
            "polarization": input_data.polarization,
        }
    else:
        npt = False
        system_arguments = {
            "topology": top,
            "nonbondedMethod": NoCutoff,
            "constraints": HBonds,
            "rigidWater": input_data.rigidWater,
            "polarization": input_data.polarization,
        }

    try:
        system = forcefield.createSystem(**system_arguments)
        use_amoeba = True
    except ValueError as e:
        if "polarization" in e.args[0]:
            system_arguments.pop("polarization")
            system = forcefield.createSystem(**system_arguments)
            use_amoeba = False

        else:
            print("Failed at create system:\n", e)
            exit()

    if npt:
        system.addForce(MonteCarloBarostat(1 * unit.atmosphere, temperature, 25))

    # we want to check if tye system has the same nonbonded exception
    # if not, we want to correct it
    if use_amoeba:
        forces = {force.__class__.__name__: force for force in system.getForces()}
        if "NonbondedForce" in list(forces.keys()):
            for ni in range(system.getNumParticles()):
                covalent15 = forces["AmoebaMultipoleForce"].getCovalentMap(
                    ni, AmoebaMultipoleForce.Covalent15
                )
                if len(covalent15) > 0:
                    q, s, e = forces["NonbondedForce"].getParticleParameters(ni)
                    for atom in covalent15:
                        if atom < ni:
                            forces["NonbondedForce"].addException(
                                particle1=ni,
                                particle2=atom,
                                chargeProd=q * unit.elementary_charge,
                                sigma=s,
                                epsilon=e,
                            )

    with open("system.xml", "w") as file:
        file.write(XmlSerializer.serialize(system))

    integrator = LangevinIntegrator(
        temperature, 1 / unit.picosecond, float(timestep) * unit.femtosecond
    )
    simulation = Simulation(
        top,
        system,
        integrator,
        Platform.getPlatformByName("CUDA"),
        {"Precision": "mixed"},  # , "DeviceIndex": deviceid},
    )
    if top.getPeriodicBoxVectors():
        simulation.context.setPeriodicBoxVectors(*top.getPeriodicBoxVectors())
    simulation.context.setPositions(positions)

    return simulation


def run(system_serialized: str, pdb_str: str, simulation_settings: SimulationSettings):
    logging.info(f"Running simulation {simulation_settings.ensemble.name}")
    cwd = os.getcwd()
    work_path = simulation_settings.work_path
    os.makedirs(work_path, exist_ok=True)
    os.chdir(work_path)

    timestep = simulation_settings.time_step.to("femtosecond").magnitude
    temperature = simulation_settings.temperature.to("kelvin").magnitude * unit.kelvin

    system = XmlSerializer.deserialize(system_serialized)
    with open("input.pdb", "w") as f:
        f.write(pdb_str)
    pdb = PDBFile("input.pdb")
    topology = pdb.topology
    position = pdb.positions

    integrator = LangevinIntegrator(
        temperature, 1 / unit.picosecond, timestep * unit.femtosecond
    )

    if simulation_settings.ensemble.name == Ensemble.NPT.name:
        logging.debug("Adding a MC Barostat")
        pressure = simulation_settings.pressure.to("bar").magnitude * unit.bar
        system.addForce(MonteCarloBarostat(pressure, temperature, 25))

    with open("system.xml", "w") as f:
        f.write(XmlSerializer.serialize(system))

    simulation = Simulation(
        topology,
        system,
        integrator,
        Platform.getPlatformByName("CUDA"),
        {"Precision": "mixed"},  # , "DeviceIndex": deviceid},
    )

    if topology.getPeriodicBoxVectors() is not None:
        simulation.context.setPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    simulation.context.setPositions(position)

    simulation.minimizeEnergy()
    # simulation.context.setVelocitiesToTemperature(temperature)

    equ_nsteps = round(1 * unit.nanosecond / (timestep * unit.femtosecond))
    try:
        simulation.reporters.append(
            StateDataReporter(
                "equilibration.csv",
                500,
                speed=True,
                volume=True,
                density=True,
                step=True,
                potentialEnergy=True,
                totalEnergy=True,
                temperature=True,
            )
        )
        # simulation.reporters.append(DCDReporter("equilibration.dcd", 500))
        simulation.step(equ_nsteps)
    except (ValueError, openmm.OpenMMException) as error:
        print(error)
        os.chdir(cwd)
        return "Failed"

    simulation.reporters.clear()
    nsteps = simulation_settings.total_steps
    # save trajectory every 100ps
    save_nsteps = round(10 * unit.picosecond / (timestep * unit.femtosecond))
    simulation.reporters.append(
        StateDataReporter(
            "simulation.csv",
            500,
            totalSteps=nsteps,
            speed=True,
            volume=True,
            density=True,
            step=True,
            potentialEnergy=True,
            totalEnergy=True,
            temperature=True,
        )
    )
    simulation.reporters.append(DCDReporter("trajectory.dcd", save_nsteps))
    simulation.reporters.append(PDBReporter("output.pdb", nsteps))
    simulation.saveState("restart.xml")
    try:
        simulation.step(nsteps)
        os.chdir(cwd)
    except (ValueError, openmm.OpenMMException) as error:
        os.chdir(cwd)
        print(error)
        return "Failed"

import os
import shutil

import openmm
from openmm import (
    LangevinIntegrator,
    MonteCarloBarostat,
    Platform,
    XmlSerializer,
    unit,
    AmoebaMultipoleForce,
)
from openmm.app import (
    PME,
    DCDReporter,
    ForceField,
    HBonds,
    PDBFile,
    NoCutoff,
    PDBReporter,
    Simulation,
    StateDataReporter,
)

# from dataclasses import dataclass
from pydantic.dataclasses import dataclass


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


def run(input_data: InputData):

    if input_data.work_dir == "None":
        work_dir = os.getcwd()
    else:
        work_dir = input_data.work_dir
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)

    ff_file = input_data.forcefield
    forcefield = ForceField(ff_file)
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
                covalentAtoms = forces["AmoebaMultipoleForce"].getCovalentMap(
                    ni, AmoebaMultipoleForce.Covalent15
                )
                if len(covalentAtoms) > 0:
                    q, s, e = forces["NonbondedForce"].getParticleParameters(ni)
                    for atom in covalentAtoms:
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
        {"Precision": "mixed", "DeviceIndex": deviceid},
    )
    if top.getPeriodicBoxVectors():
        simulation.context.setPeriodicBoxVectors(*top.getPeriodicBoxVectors())
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()

    equ_nsteps = round(1 * unit.nanosecond / (timestep * unit.femtosecond))
    try:
        simulation.step(equ_nsteps)
        simulation.reporters.clear()
    except (ValueError, openmm.OpenMMException) as error:
        print(error)
        return "Failed"

    simulation_time = input_data.simulation_time_ns
    nsteps = round(
        float(simulation_time) * unit.nanosecond / (timestep * unit.femtosecond)
    )
    # save trajectory every 100ps
    save_nsteps = round(10 * unit.picosecond / (timestep * unit.femtosecond))
    simulation.reporters.append(
        StateDataReporter(
            "simulation.log",
            save_nsteps,
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
    simulation.saveState(input_data.restart)
    try:
        simulation.step(nsteps)
    except (ValueError, openmm.OpenMMException) as error:
        print(error)
        return "Failed"

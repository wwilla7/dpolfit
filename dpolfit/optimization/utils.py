# #!/usr/bin/env python

import io
import numpy as np
import pandas as pd
import mdtraj
from enum import Enum
from dataclasses import dataclass
from typing import List, Union, Tuple, Generator, Literal
from openmm import (
    Context,
    Platform,
    LangevinIntegrator,
    XmlSerializer,
    System,
    MonteCarloBarostat,
)
from openmm.app import PDBFile
import openmm.unit as omm_unit
from dpolfit.utilities.constants import kb_u, Q_, ureg, na, kb, vacuum_permittivity
from mpid_plugin.nonbonded import (
    MPIDMultipoleHandler,
    MPIDPolarizabilityHandler,
)
from openff.interchange import Interchange
from openff.interchange.components._packmol import UNIT_CUBE, pack_box
from openff.toolkit import ForceField as OForceField
from openff.toolkit import Molecule as OMolecule
from openff.units import unit as off_unit
from collections import defaultdict, OrderedDict
from sys import stdout
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

ureg.define("alpha_unit = 1e-4/kelvin")
ureg.define("kappa_unit = 1e-6/bar")


class Ensemble(Enum):
    NVT = "NVT"
    NPT = "NPT"


@dataclass
class SimulationSettings:
    time_step: Q_ = Q_(2.0, "femtosecond")
    total_steps: int = 500000 # 1ns
    temperature: Q_ = Q_(298.15, "kelvin")
    n_moleculues: int = 256
    pressure: Q_ = Q_(1.0, "bar")
    ensemble: Ensemble = Ensemble.NPT
    work_path: str = "298_NPT"


@dataclass
class IterationRecord:
    iteration_number: int = 1
    loss_function: float = 0.0


@dataclass
class Properties:
    Density: Q_ = Q_(np.nan, "g/mL")
    DielectricConstant: Q_ = Q_(np.nan, "")
    HeatOfVaporization: Q_ = Q_(np.nan, "kJ/mol")
    ThermalExpansion: Q_ = Q_(np.nan, "alpha_unit")
    IsothermalCompressibility: Q_ = Q_(np.nan, "kappa_unit")
    GasPhaseDipole: Q_ = Q_(np.nan, "debye")
    LiquidPhaseDipole: Q_ = Q_(np.nan, "debye")
    MolecularPolarizability: Q_ = Q_(np.nan, "angstrom**3")
    Temperature: Q_ = Q_(298.15, "kelvin")


@dataclass
class SimulationOutput:
    density: Q_ = Q_(np.nan, "g/mL")
    speed: Q_ = Q_(np.nan, "ns/day")
    box_volume: Q_ = Q_(np.nan, "nm**3")
    potential_energy: Q_ = Q_(np.nan, "kJ/mol")


def read_openmm_output(
    openmm_output: str, use_last_percent: Union[int, float] = 50
) -> SimulationOutput:
    df = pd.read_csv(openmm_output, skiprows=[1])
    use_index = np.floor(len(df) * (use_last_percent / 100)).astype(int)
    use_df = df.iloc[use_index:]

    rho = use_df["Density (g/mL)"].mean(axis=0)
    speed = use_df["Speed (ns/day)"].mean(axis=0)
    box_volume = use_df["Box Volume (nm^3)"].values
    potential_energy = use_df["Potential Energy (kJ/mole)"].values

    output_data = SimulationOutput(
        density=Q_(rho, "g/mL"),
        speed=Q_(speed, "ns/day"),
        box_volume=Q_(box_volume, "nm**3"),
        potential_energy=Q_(potential_energy, "kJ/mol"),
    )

    return output_data


def create_context(
    system: System, temperature: omm_unit.Quantity = 298 * omm_unit.kelvin
) -> Context:
    # with open(system_file, "r") as f:
    #     system = XmlSerializer.deserialize(f.read())

    # MPIDForce is more stable on CUDA platform
    myplatform = Platform.getPlatformByName("CUDA")

    integrator = LangevinIntegrator(
        temperature, 1 / omm_unit.picosecond, 2.0 * omm_unit.femtosecond
    )

    context = Context(system, integrator, myplatform, {"Precision": "mixed"})
    return context


def create_simulation(
    interchange: Interchange,
    simulation_settings: SimulationSettings,
    additional_forces: List = list(),
):
    temperature = (
        simulation_settings.temperature.to("kelvin").magnitude * omm_unit.kelvin
    )
    time_step = (
        simulation_settings.time_step.to("femtosecond").magnitude * omm_unit.femtosecond
    )
    integrator = LangevinIntegrator(temperature, 1.0 / omm_unit.picosecond, time_step)
    if simulation_settings.ensemble.name == Ensemble.NPT.name:
        logging.info("Adding a MC Barostat")
        pressure = simulation_settings.pressure.to("bar").magnitude * omm_unit.bar
        barostat = MonteCarloBarostat(pressure, temperature, 25)
        additional_forces.append(barostat)

    else:
        additional_forces.clear()

    simulation = interchange.to_openmm_simulation(
        integrator=integrator, additional_forces=additional_forces
    )
    return simulation


def create_serialized_system(
    interchange: Interchange, simulation_settings: SimulationSettings
) -> Tuple[str, str]:
    system = interchange.to_openmm_system(combine_nonbonded_forces=True)
    pdb = io.StringIO()
    PDBFile.writeFile(
        topology=interchange.to_openmm_topology(),
        positions=interchange.positions.m_as("nanometer") * omm_unit.nanometer,
        file=pdb,
    )

    system_serialized = XmlSerializer.serialize(system)
    return system_serialized, pdb.getvalue()


def get_custom_parms(forcefield: OForceField) -> Tuple[OrderedDict, np.array]:

    to_custom = {
        "custom": defaultdict(OrderedDict),
        "custom_eval": defaultdict(OrderedDict),
    }
    # to_custom = defaultdict(lambda: defaultdict(lambda: defaultdict(OrderedDict)))

    parm_list = []
    for name, handler in forcefield._parameter_handlers.items():
        for parm in handler.parameters:
            if parm.attribute_is_cosmetic("custom"):
                this_dict = to_custom["custom"][name].setdefault(
                    parm.smirks, OrderedDict()
                )
                custom_parameters = getattr(parm, "_custom").split()
                for k in custom_parameters:
                    this_k = getattr(parm, k)
                    this_dict[k] = this_k
                    parm_list.append(this_k.magnitude)

            if parm.attribute_is_cosmetic("custom_eval"):
                this_dict = to_custom["custom_eval"][name].setdefault(
                    parm.smirks, OrderedDict()
                )
                custom_eval_parameters = getattr(parm, "_custom_eval").split()
                for k in custom_eval_parameters:
                    tmp = k.split("/")
                    this_dict[tmp[0]] = tmp[1:]
    return to_custom, np.array(parm_list)


def set_custom_parms(
    forcefield: OForceField, to_custom: OrderedDict, parm_list: np.array
) -> OForceField:
    count = 0
    for k, v in to_custom["custom"].items():
        handler = forcefield.get_parameter_handler(k)
        for smirk, parameters in v.items():
            parm = handler.parameters[smirk]
            for name, value in parameters.items():
                setattr(parm, name, parm_list[count] * value.units)
                count += 1

    for k, v in to_custom["custom_eval"].items():
        handler = forcefield.get_parameter_handler(k)
        for smirk, parameters in v.items():
            for name, expression in parameters.items():
                parm = getattr(handler[expression[0]], expression[1])
                new_parm = eval(f"{expression[2]}{parm.magnitude}") * parm.units
                this_parm = handler.parameters[smirk]
                setattr(this_parm, name, new_parm)

    return forcefield


def compute_hvap_alpha_kappa(
    gas_mdLog: SimulationOutput,
    liquid_mdLog: SimulationOutput,
    liquid_mdSettings: SimulationSettings,
    calc_hvap=False,
) -> Properties:

    box_volume = liquid_mdLog.box_volume.to("nm**3")
    box_volume_mean = box_volume.mean()
    temperature = liquid_mdSettings.temperature.to("kelvin")
    # H = E + PV
    PV_liquid = (
        liquid_mdSettings.pressure.to("bar")
        * box_volume
        / (liquid_mdSettings.n_moleculues / na.to_base_units())
    )

    H_liquid = liquid_mdLog.potential_energy.to("kJ/mol") + PV_liquid
    H_liquid_mean = H_liquid.mean()

    kbT = kb_u * temperature

    if calc_hvap:

        hvap = (
            gas_mdLog.potential_energy.to("kJ/mol").mean(axis=0)
            + kbT
            - H_liquid_mean / liquid_mdSettings.n_moleculues
        )

    else:
        hvap = Q_(np.nan, "kJ/mole")

    alpha = (
        ((H_liquid * box_volume).mean(axis=0) - H_liquid_mean * box_volume_mean)
        / box_volume_mean
        / kb_u
        / temperature
        / temperature
    )

    kappa = (
        (1 / kb / temperature)
        * ((box_volume**2).mean() - box_volume_mean**2)
        / box_volume_mean
    )

    ret = Properties(
        Density=Q_(liquid_mdLog.density, "g/mL"),
        HeatOfVaporization=hvap,
        ThermalExpansion=alpha.to("alpha_unit"),
        IsothermalCompressibility=kappa.to("kappa_unit"),
    )

    return ret


def _getChargesPolarizabilities(force, n_particles) -> Tuple[np.array, np.array]:
    chargs = np.zeros(n_particles)
    polarizabilities = np.zeros(n_particles)
    for p in range(n_particles):
        parms = force.getMultipoleParameters(p)
        chargs[p] = parms[0]
        polarizabilities[p] = parms[-1][0]

    return chargs, polarizabilities


def _getPermanetDipoles(positions, charges) -> Q_:
    """
    positions: mdtraj xyz, unit in nanometer
    charges: shape (natoms)
    """

    dipole_moments = np.dot(positions.transpose(0, 2, 1), charges)
    # avg_dipole_moments = Q_(np.linalg.norm(dipole_moments, axis=1).mean(), "e*nm").to(
    #     "debye"
    # )

    return Q_(dipole_moments, "e*nm")


def _getResidueDipoles(
    residues: Generator,
    PermanentDipoles: np.array,
    InducedDipoles: np.array,
) -> Q_:
    """
    positions: mdtraj xyz, unit in nanometer
    charges: shape (natoms)
    """
    permanent_dipole_norm = np.linalg.norm(PermanentDipoles)
    data = []
    for res in residues:
        atom_indices = [a.index for a in res.atoms()]
        induced_dipoles = InducedDipoles[atom_indices].sum(axis=0)
        total_norm = permanent_dipole_norm + np.linalg.norm(induced_dipoles)
        # total = PermanentDipoles + induced_dipoles
        # total_norm = np.linalg.norm(total)

    data.append(total_norm)
    average_dipoles = Q_(np.mean(data), "nm*e").to("debye")

    return average_dipoles


def compute_DielectricProperties(
    system: System,
    topology_file: str,
    trajectory_file: str,
    use_last_percent: int = 50,
    temperature: omm_unit.Quantity = 298.15 * omm_unit.kelvin,
    MPID: bool = True,
):
    context = create_context(system=system, temperature=temperature)
    n_particles = system.getNumParticles()
    if MPID:
        import mpidplugin
        from mpidplugin import MPIDForce

        for force in system.getForces():
            if MPIDForce.isinstance(force):
                mpidforce = MPIDForce.cast(force)

    else:
        from openmm import NonbondedForce

        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                charges = np.array(
                    [
                        force.getParticleParameters(i)[0] / omm_unit.elementary_charge
                        for i in range(n_particles)
                    ]
                )

    all_traj = mdtraj.load_dcd(trajectory_file, top=topology_file)
    n_frames = all_traj.n_frames

    index = np.floor(n_frames * (use_last_percent / 100)).astype(int)
    traj = all_traj[index:]
    avg_volumes = Q_(
        np.mean(
            list(
                map(
                    np.linalg.det,
                    [traj.openmm_boxes(i) / omm_unit.bohr for i in range(index)],
                )
            )
        ),
        "a0**3",
    )

    dipole_moments = np.zeros((index, 3))
    induced_dipoles = np.zeros((index, n_particles, 3))
    if MPID:
        for f in range(index):
            box = traj.openmm_boxes(frame=f)
            position = traj.openmm_positions(frame=f)
            context.setPeriodicBoxVectors(*box)
            context.setPositions(position)
            dipole_moments[f] = (
                Q_(mpidforce.getSystemMultipoleMoments(context)[1:4], "debye")
                .to("e*a0")
                .magnitude
            )
            induced_dipoles[f] = np.array(mpidforce.getInducedDipoles(context))

    else:
        dipole_moments = _getPermanetDipoles(traj.xyz, charges)

    ### fluctuation of dipole moments
    avg_sqr_mus_au = np.mean(np.square(dipole_moments), axis=0)
    avg_mus_sqr_au = np.square(np.mean(dipole_moments, axis=0))

    variance = Q_(np.sum(avg_sqr_mus_au - avg_mus_sqr_au), "e**2*a0**2")

    prefactor = Q_(
        1
        / (
            3
            * vacuum_permittivity
            * kb.to("J/kelvin")
            * Q_(temperature / omm_unit.kelvin, "kelvin")
        )
        .to("hartree")
        .magnitude,
        "a0/e**2",
    )

    prefactor2 = 1 / vacuum_permittivity

    charges, polarizabilities = _getChargesPolarizabilities(mpidforce, n_particles)

    sum_alphas = Q_(np.sum(polarizabilities), "nm**3").to("a0**3")
    dielectric = prefactor * variance / avg_volumes
    high_frequency_dielectric = prefactor2 * (1 / avg_volumes) * sum_alphas + 1

    dielectric_constant = dielectric + high_frequency_dielectric

    # Residue Dipole
    pdb = PDBFile(topology_file)
    residues = pdb.topology.residues()
    random_frame = np.random.randint(1, index)
    n_atom_per_residue = int(pdb.topology.getNumAtoms() / pdb.topology.getNumResidues())
    gas_phase = _getPermanetDipoles(
        positions=np.array(
            (pdb.positions / omm_unit.nanometer)[:n_atom_per_residue]
        ).reshape(1, n_atom_per_residue, 3),
        charges=charges[:n_atom_per_residue],
    )
    condensed_phase = _getResidueDipoles(
        residues, gas_phase.magnitude, induced_dipoles[random_frame]
    )
    
    molecular_polarizability = Q_(sum(polarizabilities[:n_atom_per_residue]), "nm**3").to("angstrom**3")

    ret = Properties(
        DielectricConstant=dielectric_constant,
        LiquidPhaseDipole=condensed_phase.to("debye"),
        GasPhaseDipole=Q_(np.linalg.norm(gas_phase), "e*nm").to("debye"),
        MolecularPolarizability=molecular_polarizability,
    )

    return ret


def compute(
    gas_mdLog: SimulationOutput,
    liquid_mdLog: SimulationOutput,
    liquid_mdSettings: SimulationSettings,
    system_file: str,
    topology_file: str,
    trajectory_file: str,
    use_last_percent: int = 50,
    calc_hvap=True,
    MPID: bool = True,
):
    with open(system_file, "r") as f:
        system = XmlSerializer.deserialize(f.read())

    temperature = liquid_mdSettings.temperature.to("kelvin").magnitude * omm_unit.kelvin
    p1 = compute_hvap_alpha_kappa(
        gas_mdLog, liquid_mdLog, liquid_mdSettings, calc_hvap=True
    )
    p2 = compute_DielectricProperties(
        system=system,
        topology_file=topology_file,
        trajectory_file=trajectory_file,
        use_last_percent=use_last_percent,
        temperature=temperature,
        MPID=True,
    )
    p2.Density = p1.Density
    p2.HeatOfVaporization = p1.HeatOfVaporization
    p2.ThermalExpansion = p1.ThermalExpansion
    p2.IsothermalCompressibility = p1.IsothermalCompressibility

    return p2

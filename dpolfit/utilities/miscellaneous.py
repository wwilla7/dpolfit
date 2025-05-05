#!/usr/bin/env python

from openff.toolkit.topology import Molecule
from openff.units import unit
from qcelemental.models.molecule import Molecule as qcMolecule
from openmm.app import PDBFile, Residue, Chain, Topology
from openmm import unit as omm_unit
import numpy as np

try:
    from openeye import oechem
    from openeye.oechem import OEBlobType, OEField, OEMolRecord, OEStringType

    mapped_smile_field = OEField(
        "canonical_isomeric_explicit_hydrogen_mapped_smiles", OEStringType
    )
    angstrom_xyz_field = OEField("angstrom_xyz", OEStringType)
    psi4_xyz_field = OEField("psi4_xyz", OEStringType)
    geometry_angstrom_field = OEField("geometry_angstrom", OEStringType)
    grid_angstrom_field = OEField("grid_angstrom", OEStringType)
    grid_esp_0_au_field = OEField("grid_esp_0_au", OEStringType)

    smiTags = oechem.OESMILESFlag_AtomMaps
    use_openeye = True
except ImportError as e:
    print(e)
    use_openeye = False

perturb_dipoles = {
    "0": [0.0, 0.0, 0.0],
    "x+": [0.01, 0.0, 0.0],
    "x-": [-0.01, 0.0, 0.0],
    "y+": [0.0, 0.01, 0.0],
    "y-": [0.0, -0.01, 0.0],
    "z+": [0.0, 0.0, 0.01],
    "z-": [0.0, 0.0, -0.01],
}

if use_openeye:

    def _qcmol2oemol(qcmol: qcMolecule) -> OEMolRecord:
        """
        Convert QCMol to OEMol

        Convert a QM Molecule to an OE Molecule and store all information
        on an OEMolRecord.

        Parameters
        ----------
        qcmol : qcMolecule
                QC molecule from the QCArchive records

        Returns
        -------
        OEMolRecord
            Returned OEMolRecord
        """
        oemolrecord = OEMolRecord()
        mapped_smi = qcmol.extras["canonical_isomeric_explicit_hydrogen_mapped_smiles"]
        geometry = qcmol.geometry * unit.bohr
        # symbols = qcmol.symbols
        bohr_xyz = qcmol.to_string("psi4")
        angstrom_xyz = qcmol.to_string("xyz")

        # generate an offmol
        offmol = Molecule.from_mapped_smiles(
            mapped_smiles=mapped_smi, allow_undefined_stereo=True
        )
        offmol.generate_conformers(n_conformers=1)
        offmol.conformers.clear()
        offmol.conformers.append(geometry)

        # convert to an oemol
        oemol = offmol.to_openeye()
        oemolrecord.set_mol(oemol)
        oemolrecord.set_value(mapped_smile_field, mapped_smi)
        oeconf = oemol.GetConfs().next()

        # set values on conf records
        conf_record = oemolrecord.get_conf_record(oeconf)
        conf_record.set_value(psi4_xyz_field, bohr_xyz)
        conf_record.set_value(angstrom_xyz_field, angstrom_xyz)
        conf_record.set_value(mapped_smile_field, mapped_smi)

        oemolrecord.set_conf_record(oeconf, conf_record)
        oemolrecord.set_mol(oemol)

        return oemolrecord


def create_monomer(pdb_file: str, output_file: str):
    pdb = PDBFile(pdb_file)
    top = pdb.topology
    pos = pdb.positions

    residues = [r for r in top.residues()]
    new_top = Topology()
    new_chain = new_top.addChain()
    new_res = new_top.addResidue(name=residues[0].name, chain=new_chain, id=0)

    indices = []
    for a in residues[0].atoms():
        new_top.addAtom(name=a.name, element=a.element, residue=new_res, id=a.id)
        indices.append(int(a.id))

    new_pos = pos[indices[0] : indices[-1] + 1]

    PDBFile.writeFile(new_top, new_pos, open(output_file, "w"))

    return output_file


def remove_unit_for_xml(v):
    if isinstance(v, omm_unit.quantity.Quantity):
        return str(v.value_in_unit(v.unit))
    elif isinstance(v, unit.Quantity):
        return str(v.m_as(v.units))
    elif isinstance(v, float):
        return str(v)
    elif isinstance(v, int):
        return str(v)
    else:
        return v


def get_angle(a, b, c):
    ret = np.arccos((c**2 - a**2 - b**2) / (-2 * a * b)) * 180 / np.pi
    return ret


def get_bond_length(a, b, theta):
    theta = theta * np.pi / 180
    ret = np.sqrt(a**2 + b**2 - 2 * a * b * np.cos(theta))
    return ret


def get_length_oh(theta, c):
    theta = theta * np.pi / 180
    ret = np.sqrt(c**2 / 2 / (1 - np.cos(theta)))
    return ret

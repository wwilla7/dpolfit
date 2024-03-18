#!/usr/bin/env python

import pickle
from openeye import oechem
from oechem import OEMolRecord, OEField, OEStringType, OEBlobType
from openff.toolkit.topology import Molecule
from openff.units import unit
from qcelemental.models.molecule import Molecule as qcMolecule

mapped_smile_field = OEField(
    "canonical_isomeric_explicit_hydrogen_mapped_smiles", OEStringType
)
angstrom_xyz_field = OEField("angstrom_xyz", OEStringType)
psi4_xyz_field = OEField("psi4_xyz", OEStringType)
geometry_angstrom_field = OEField("geometry_angstrom", OEStringType)
grid_angstrom_field = OEField("grid_angstrom", OEStringType)
grid_esp_0_au_field = OEField("grid_esp_0_au", OEStringType)

perturb_dipoles = {
    "0":  [0.0, 0.0, 0.0],
    "x+": [0.01, 0.0, 0.0],
    "x-": [-0.01, 0.0, 0.0],
    "y+": [0.0, 0.01, 0.0],
    "y-": [0.0, -0.01, 0.0],
    "z+": [0.0, 0.0, 0.01],
    "z-": [0.0, 0.0, -0.01],
}


def qcmol2oemol(qcmol: qcMolecule) -> OEMolRecord:
    """
        Convert QCMol to OEMol

        Convert a QM Molecule to an OE Molecule and store all information
        on an OEMolRecord.

        Parameters
        ----------
        qcmol : qcMolecule
    coordinates.npy        QC molecule from the QCArchive records

        ROEMolRecordeturns
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


# testing

if __name__ == "__main__":
    qcmol = pickle.load(open("test_mol.pkl", "rb"))

    ofs = oechem.oeofstream("test_mol.oedb")

    ret = qcmol2oemol(qcmol)

    oechem.OEWriteRecord(ofs, ret)

    print(ret)

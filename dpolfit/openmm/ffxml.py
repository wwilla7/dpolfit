#!/usr/bin/env python
"""
This module contains functions to customize openmm xml force field files.
"""

import os

# from pprint import pprint
from collections import defaultdict
from datetime import datetime

import numpy as np
from dpolfit.fitting.respdpol import assign_polarizability
from dpolfit.utilities.constants import a03_to_nm3
from dpolfit.utilities.miscellaneous import remove_unit_for_xml
from lxml import etree
from openeye import oechem
from openff.toolkit import ForceField, Molecule

from openmm import (
    AmoebaMultipoleForce,
    HarmonicAngleForce,
    HarmonicBondForce,
    NonbondedForce,
    PeriodicTorsionForce,
)


def get_ff_structure(atoms: list, residue: list):
    forcefield_data = {
        "AtomTypes": {"tags": ["Type"], "parameters": atoms, "attrs": {}},
        "Residues": {
            "tags": ["Residue", "Atom", "Bond"],
            "parameters": residue,
            "attrs": {},
        },
        "HarmonicBondForce": {"tags": ["Bond"], "parameters": [], "attrs": {}},
        "HarmonicAngleForce": {"tags": ["Angle"], "parameters": [], "attrs": {}},
        "PeriodicTorsionForce": {"tags": ["Proper"], "parameters": [], "attrs": {}},
        "NonbondedForce": {
            "tags": ["Atom"],
            "parameters": [],
            "attrs": {"coulomb14scale": "0.8333333333", "lj14scale": "0.5"},
        },
        # "MPIDForce": {"tags": ["Multipole", "Polarize"], "parameters": {"multipole": [], "polarize": []},
        #               "attrs": {"coulomb14scale": "1.0"},},}
        "AmoebaMultipoleForce": {
            "tags": ["Multipole", "Polarize"],
            "parameters": {"multipole": [], "polarize": []},
            "attrs": {
                "direct11Scale": "0.0",
                "direct12Scale": "0.0",
                "direct13Scale": "0.0",
                "direct14Scale": "0.0",
                "mpole12Scale": "0.0",
                "mpole13Scale": "0.0",
                "mpole14Scale": "0.5",
                "mpole15Scale": "1.0",
                "mutual11Scale": "0.0",
                "mutual12Scale": "0.0",
                "mutual13Scale": "0.0",
                "mutual14Scale": "0.0",
                "polar12Scale": "0.0",
                "polar13Scale": "0.0",
                "polar14Intra": "0.0",
                "polar14Scale": "0.0",
                "polar15Scale": "0.0",
            },
        },
    }
    return forcefield_data


def get_ff_data(ff_file: str) -> dict:
    """
    [TODO:description]

    :param ff_file: [TODO:description]
    :type ff_file: str
    :return: [TODO:description]
    :rtype: dict
    """
    tree = etree.parse(ff_file)
    root = tree.getroot()

    data = {}
    for force in root:
        data[force.tag] = {}
        data[force.tag]["attrib"] = force.attrib
        data[force.tag]["parameters"] = []
        for atom in force:
            if force.tag == "Residues":
                data[force.tag]["parameters"].append(
                    {"Residue": {"attrib": dict(atom.attrib), "parameters": []}}
                )
                for r in atom:
                    data[force.tag]["parameters"][0]["Residue"]["parameters"].append(
                        {r.tag: dict(r.attrib)}
                    )
            else:
                data[force.tag]["parameters"].append({atom.tag: dict(atom.attrib)})

    data["Info"]["parameters"][0]["DateGenerated"] = datetime.now().strftime(
        "%m-%d-%Y %H:%M:%S"
    )

    return data


def write_ff_xml(data: dict) -> str:
    """
    [TODO:description]

    :param data: [TODO:description]
    :type data: dict
    :return: [TODO:description]
    :rtype: str
    """

    root = etree.Element("ForceField")
    for force, items in data.items():
        child = etree.SubElement(root, force, **items["attrib"])
        for item in items["parameters"]:
            for k, vs in item.items():
                if k == "Residue":
                    res = etree.SubElement(child, k, **vs["attrib"])
                    for p in vs["parameters"]:
                        for a, b in p.items():
                            ret = etree.SubElement(res, a, **b)
                else:
                    if isinstance(vs, dict):
                        data = etree.SubElement(child, k, **vs)
                    else:
                        data = etree.SubElement(child, k)
                        data.text = f'{datetime.now().strftime("%m-%d-%Y %H:%M:%S")} on {os.uname().nodename}'

    tree = etree.ElementTree(root)
    etree.indent(tree, "  ")
    data = etree.tostring(tree, encoding="utf-8", pretty_print=True).decode("utf-8")

    return data


def get_pgrp(oemol: oechem.OEMol) -> (dict, dict):
    """
    Function to define polarization group based on rotatable bonds

    :param oemol: input molecule
    :type oemol: oechem.OEMol
    :return: polarization groups and atom type maps.
    :rtype: (dict, dict)
    """

    oechem.OEPerceiveSymmetry(oemol, includeH=True)
    pairs = defaultdict(list)
    for atom in oemol.GetAtoms():
        pairs[atom.GetSymmetryClass()].append(atom)
    typemaps = {}
    for idx, (_, pair) in enumerate(pairs.items()):
        for atom in pair:
            typemaps[atom.GetIdx()] = {
                k: f"4{idx+1:02d}" for k in ["name", "class", "type"]
            } | {"rname": f"4{atom.GetIdx()+1:02d}"}

    bonds = defaultdict(list)
    oechem.OEFindRingAtomsAndBonds(oemol)
    for bond in oemol.GetBonds():
        group = False
        ba = bond.GetBgn()
        ea = bond.GetEnd()
        if not bond.IsRotor():
            group = True
            if ba.GetAtomicNum() == ea.GetAtomicNum():
                group = False
        else:
            if ba.GetAtomicNum() == ea.GetAtomicNum():
                group = True

        if group:
            b = bond.GetBgnIdx()
            e = bond.GetEndIdx()
            bonds[b].append(e)
            bonds[e].append(b)

    pgrp = defaultdict(set)
    for b, es in bonds.items():
        for a in es:
            pgrp[typemaps[b]["type"]].add(typemaps[a]["type"])

    return pgrp, typemaps


def create_forcefield(
    polarizability: dict,
    pol_unit: str = "a0**3",
    mol2_file: str = "molecule.mol2",
    openff: str = "openff_unconstrained-2.2.0.offxml",
) -> str:
    """
    Function to create force field to use with OpenMM

    :param polarizability: a dictionary contains polarizability parameters
    :type polarizability: dict
    :param pol_unit: polarizability unit, defaults to "a0**3"
    :type pol_unit: str, optional
    :param mol2_file: molecule file that contains partial charges, defaults to "molecule.mol2"
    :type mol2_file: str, optional
    :param openff: the open force field vision to obtain valence terms, defaults to "openff_unconstrained-2.2.0.offxml"
    :type openff: str, optional
    :return: force field xml file to create a OpenMM system
    :rtype: str
    """
    offmol = Molecule.from_file(mol2_file, file_format="mol2")
    top = offmol.to_topology()
    forcefield = ForceField(openff)
    system = forcefield.create_openmm_system(
        topology=top, charge_from_molecules=[offmol]
    )
    n_particles = system.getNumParticles()

    oemol = offmol.to_openeye()
    oemol = assign_polarizability(oemol, polarizability, pol_unit)
    pgrp, omm_atom_maps = get_pgrp(oemol)

    omm_top = top.to_openmm()

    pol_dict = {
        atom.GetIdx(): atom.GetData("polarizability") * a03_to_nm3
        for atom in oemol.GetAtoms()
    }

    atoms = []
    residue = {"residue": [{"name": "MOL"}], "atom": [], "bond": []}

    for a in omm_top.atoms():
        a.name = omm_atom_maps[a.index]["name"]
        ret = {
            "name": omm_atom_maps[a.index]["name"],
            "class": omm_atom_maps[a.index]["class"],
            "element": a.element.symbol,
            "mass": remove_unit_for_xml(a.element.mass),
        }
        if ret in atoms:
            pass
        else:
            atoms.append(ret)
        residue["atom"].append(
            {
                "name": omm_atom_maps[a.index]["rname"],
                "type": omm_atom_maps[a.index]["type"],
            }
        )

    for b in omm_top.bonds():
        ret = {
            "atomName1": omm_atom_maps[b[0].index]["rname"],
            "atomName2": omm_atom_maps[b[1].index]["rname"],
        }
        residue["bond"].append(ret)

    ff_data = get_ff_structure(atoms, residue)

    for force in system.getForces():
        force_name = force.__class__.__name__
        if isinstance(force, NonbondedForce):
            keys = ["charge", "sigma", "epsilon"]
            for ni in range(n_particles):
                ret = force.getParticleParameters(ni)
                q = ret[0]
                # set charge in nonbonded force to zeros
                ret[0] = "0.0"
                this_param = ff_data[force_name]["parameters"]
                data = {"class": omm_atom_maps[ni]["class"]} | {
                    k: remove_unit_for_xml(v) for k, v in zip(keys, ret)
                }

                if data in this_param:
                    pass
                else:
                    this_param.append(data)
                multipole = {
                    "type": omm_atom_maps[ni]["type"],
                    "kz": "0",
                    "kx": "0",
                    "c0": remove_unit_for_xml(q),
                    "q11": "0.0",
                }

                multipole |= {f"d{i+1}": "0.0" for i in range(3)}
                multipole |= {f"q2{i+1}": "0.0" for i in range(2)}
                multipole |= {f"q3{i+1}": "0.0" for i in range(3)}

                this_param = ff_data["AmoebaMultipoleForce"]["parameters"]["multipole"]
                if multipole in this_param:
                    pass
                else:
                    this_param.append(multipole)

                this_param = ff_data["AmoebaMultipoleForce"]["parameters"]["polarize"]

                pols = {
                    "type": omm_atom_maps[ni]["type"],
                    "polarizability": np.round(pol_dict[ni], 9).astype(str),
                    "thole": "0.0",
                }
                pols |= {
                    f"pgrp{i+1}": v
                    for i, v in enumerate(pgrp[omm_atom_maps[ni]["type"]])
                }

                if pols in this_param:
                    pass
                else:
                    this_param.append(pols)

        elif isinstance(force, PeriodicTorsionForce):
            torsions = {}
            n_periodicity = 1
            for ni in range(n_particles):
                ret = force.getTorsionParameters(ni)
                particles = str(ret[:4])
                if particles in list(torsions.keys()):
                    n_periodicity += 1
                else:
                    n_periodicity = 1
                    params = {
                        f"class{i + 1}": omm_atom_maps[ret[i]]["class"]
                        for i in range(4)
                    }
                    torsions[particles] = []
                    torsions[particles].append(params)

                keys = [
                    f"periodicity{n_periodicity}",
                    f"phase{n_periodicity}",
                    f"k{n_periodicity}",
                ]
                torsions[particles].append(
                    {k: remove_unit_for_xml(v) for k, v in zip(keys, ret[4::])}
                )

            this_param = ff_data[force_name]["parameters"]

            for k, v in torsions.items():
                combined = {}
                [combined.update(v0) for v0 in v]
                if combined in this_param:
                    pass
                else:
                    this_param.append(combined)

        elif isinstance(force, HarmonicAngleForce):
            keys = ["class1", "class2", "class3", "angle", "k"]
            for ni in range(force.getNumAngles()):
                ret = force.getAngleParameters(ni)
                params = {
                    k: (omm_atom_maps[v]["class"] if i < 3 else remove_unit_for_xml(v))
                    for i, (k, v) in enumerate(zip(keys, ret))
                }
                this_param = ff_data[force_name]["parameters"]
                if params in this_param:
                    pass
                else:
                    this_param.append(params)

        elif isinstance(force, HarmonicBondForce):
            keys = ["class1", "class2", "length", "k"]
            for ni in range(force.getNumBonds()):
                ret = force.getBondParameters(ni)
                params = {
                    k: (omm_atom_maps[v]["class"] if i < 2 else remove_unit_for_xml(v))
                    for i, (k, v) in enumerate(zip(keys, ret))
                }
                this_param = ff_data[force_name]["parameters"]
                if params in this_param:
                    pass
                else:
                    this_param.append(params)

    ######################################
    ########### File Structure ###########
    tree = etree.Element("ForceField")
    ############ Info ####################
    info = etree.SubElement(tree, "Info")
    today = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
    date = etree.SubElement(info, "DateGenerated")
    date.text = today
    author = etree.SubElement(info, "GeneratedBy")
    author.text = "Liangyue Willa Wang"
    ############ Info ####################

    for force, v in ff_data.items():
        tags = v["tags"]
        attrs = v["attrs"]
        parent = etree.SubElement(tree, force, **attrs)

        if len(tags) > 1:
            for tag in tags:
                for param in v["parameters"][tag.lower()]:
                    if tag == "Residue":
                        parent = etree.SubElement(parent, tag, **param)
                    else:
                        child = etree.SubElement(parent, tag, **param)
        else:
            for param in v["parameters"]:
                child = etree.SubElement(parent, tags[0], **param)

    etree.indent(tree, "  ")
    ff_string = etree.tostring(tree, encoding="utf-8", pretty_print=True).decode(
        "utf-8"
    )

    return ff_string

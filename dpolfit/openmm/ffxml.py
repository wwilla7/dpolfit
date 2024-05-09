#!/usr/bin/env python
"""
This module contains functions to customize openmm xml force field files.
"""

import os
from datetime import datetime
#from pprint import pprint

import numpy as np
from lxml import etree

from openff.toolkit import ForceField, Molecule
from dpolfit.fitting.respdpol import assign_polarizability
from dpolfit.utilities.constants import a03_to_nm3
from dpolfit.utilities.miscellaneous import remove_unit_for_xml
from openmm import (
    AmoebaMultipoleForce,
    NonbondedForce,
    HarmonicAngleForce,
    HarmonicBondForce,
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


def update_ff(ff_file: str, parameters: dict) -> str:
    """
    [TODO:description]

    :param ff_file: [TODO:description]
    :type ff_file: str
    :param parameters: [TODO:description]
    :type parameters: dict
    :return: [TODO:description]
    :rtype: str
    """
    tree = etree.parse(ff_file)
    root = tree.getroot()

    for k, v in parameters.items():
        ret = root.findall(k)
        for vn, vv in v.items():
            for r in ret:
                r.set(vn, str(vv))

    q1 = root.findall("./AmoebaMultipoleForce/Multipole[@type='401']")
    q2 = root.findall("./AmoebaMultipoleForce/Multipole[@type='402']")
    q2[0].set("c0", str(-0.5 * float(q1[0].attrib["c0"])))

    d = root.find(".//DateGenerated")
    d.text = f'{datetime.now().strftime("%m-%d-%Y %H:%M:%S")} on {os.uname().nodename}'

    etree.indent(tree, "  ")
    ret = etree.tostring(tree, encoding="utf-8", pretty_print=True).decode("utf-8")

    return ret


def update_results(input_array: np.array, parameters: dict) -> dict:
    """
    Update the parameters with new data split out from the optimizer

    # tests:
    # data = json.load(open("updates.json", "r"))
    # ret = update_from_template(np.random.rand(7), data)
    # print(ret)

    :param input_array: [TODO:description]
    :type input_array: np.array
    :param parameters: [TODO:description]
    :type parameters: dict
    :return: [TODO:description]
    :rtype: dict
    """

    data = input_array.tolist()

    for k, v in parameters.items():
        for vn in v:
            v[vn] = data.pop(0)

    return parameters


def _get_input(ff_file: str, parameters: dict) -> dict:
    """
    [TODO:description]

    :param ff_file: [TODO:description]
    :type ff_file: str
    :param parameters: [TODO:description]
    :type parameters: dict
    :return: [TODO:description]
    :rtype: dict
    """

    tree = etree.parse(ff_file)
    root = tree.getroot()

    for k, v in parameters.items():
        ret = root.findall(k)
        for vn, vv in v.items():
            for r in ret:
                dt = r.get(vn)
                v[vn] = dt
                print(dt)

    return parameters


def create_forcefield(
    polarizability: dict,
    pol_unit: str = "a0**3",
    mol2_file: str = "molecule.mol2",
    openff: str = "openff_unconstrained-2.2.0.offxml",
) -> str:
    offmol = Molecule.from_file(mol2_file, file_format="mol2")
    top = offmol.to_topology()
    forcefield = ForceField(openff)
    system = forcefield.create_openmm_system(
        topology=top, charge_from_molecules=[offmol]
    )
    n_particles = system.getNumParticles()

    omm_atom_maps = {}
    charges = {}
    type_id = 0
    for idx, q in enumerate(offmol.partial_charges):
        if q in list(charges.keys()):
            this_id = charges[q]
        else:
            this_id = f"7{type_id:02d}"
            charges[q] = this_id
            type_id += 1

        omm_atom_maps[idx] = {
            k: this_id for k in ["name", "class", "type"]
        } | {"rname": f"7{idx+1:02d}"}

    omm_top = top.to_openmm()

    oemol = offmol.to_openeye()
    oemol = assign_polarizability(oemol, polarizability, pol_unit)

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
                    "thole": "0",
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
    author.text = "Willa Wang"
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

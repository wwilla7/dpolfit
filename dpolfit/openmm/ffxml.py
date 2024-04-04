#!/usr/bin/env python
"""
This module contains functions to customize openmm xml force field files.
"""

import os
from datetime import datetime

import numpy as np
from lxml import etree


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

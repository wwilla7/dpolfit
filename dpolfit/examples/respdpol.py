"""
This file contains an example to fit respdpol stype partial charges
with typed polarizabilities.
"""

import os
from dpolfit.fitting.respdpol import fit
import json
from openeye import oechem
import importlib_resources

data_path = importlib_resources.files("dpolfit")

trainingset = os.path.join(data_path, "data", "tests", "dataset.oedb")
paramerters = os.path.join(data_path, "examples", "polarizabilities.json")

ifs = oechem.oeifstream(trainingset)

data = json.load(open(paramerters, "r"))

for oerecord in oechem.read_mol_records(ifs):
    mol = oerecord.get_mol()
    print("SMILES", oechem.OEMolToSmiles(mol))
    ret = fit(
        oerecord=oerecord, polarizabilities=data["parameters"], unit=data["_unit"]
    )
    for k, v in zip(["resp1", "resp2", "rrms"], ret):
        print("{0:7}{1}".format(k, v))

    print("-" * 25)

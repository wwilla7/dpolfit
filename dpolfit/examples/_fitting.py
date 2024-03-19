#!/usr/bin/env python

"""
This file provide a script to derive typed polarizabilities.

"""

import json
import os

import importlib_resources
import ray

from dpolfit import element_typed
from dpolfit.fitting.polarizability import train
from dpolfit.psi4.qmesps import worker


def main(**args):
    """
    An example to derive typed polarizability parameters from quantum chemistry
    """

    data_path = importlib_resources.files("dpolfit").joinpath("data")

    # ray is used to distribute QM calcs
    ray_tmp_path = "/tmp/ray_tmp"
    os.makedirs(ray_tmp_path, exist_ok=True)
    ray.init(_temp_dir=ray_tmp_path, num_cpus=16)

    wd = worker(
        input_file=os.path.join(data_path, "input.smi"),
        wd=os.path.join(data_path, "tests"),
    )

    ray.shutdown()

    trainingset = os.path.join(wd, "dataset.oedb")
    pol_ret = train(trainingset, parameter_types=element_typed)
    pol_ret |= {"_trainingset": trainingset}
    json.dump(pol_ret, open("polarizabilities.json", "w"), indent=2)

    print("-" * 22)
    for k, v in pol_ret.items():
        print("{0:22}|  {1}".format(k, v))


if __name__ == "__main__":
    main()

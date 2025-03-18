#!/usr/bin/env python
"""
This module contains some conversion constants.
"""

import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

a03_to_angstrom3 = Q_(1, "bohr**3").to("angstrom**3").magnitude
a03_to_nm3 = Q_(1, "bohr**3").to("nm**3").magnitude
a03_to_ang3 = Q_(1, "bohr").to("angstrom").magnitude  # 0.14818471147298395

degree_to_radian = Q_(1, "degree").to("radian").magnitude  # 0.017453292519943295

kcal_to_kj = Q_(1, "kcal").to("kJ").magnitude  # 4.184
kb = Q_(1, ureg.boltzmann_constant).to("kJ/kelvin")
na = Q_(1, "N_A")
kb_u = (kb / (1 / na).to("mole")).to("kJ/kelvin/mole")

opc3Epol = (
    Q_((2.43 - 1.855), "debye").to("e*angstrom") ** 2 / (2 * Q_(1.44, "angstrom**3"))
).to(
    "e**2/a0"
)  # hartree
opc3Epolkjmol = Q_(opc3Epol.magnitude, "hartree").to("kJ") / (1 / na).to("mole")

vacuum_permittivity = (
    Q_(8.854187812e-12, "F/m").to("e**2/a0/hartree").magnitude
)  # this is unitless

j_per_cm_square_to_mPa = 1e9
D0_constant = 2.837297
kbt298_in_joule = 4.11e21

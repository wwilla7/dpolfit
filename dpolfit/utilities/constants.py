import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

a03_to_angstrom3 = Q_(1, "bohr**3").to("angstrom**3").magnitude
a03_to_nm3 = Q_(1, "bohr**3").to("nm**3").magnitude
a03_to_ang3 = Q_(1, "bohr").to("angstrom").magnitude # 0.14818471147298395


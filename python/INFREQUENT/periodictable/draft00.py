import periodictable

periodictable.hydrogen.mass
periodictable.hydrogen.mass_units
periodictable.Cd.density
periodictable.Cd.density_units

periodictable.B.neutron.absorption
# Ni f1/f2 for Cu K-alpha X-rays
periodictable.Ni.xray.scattering_factors(wavelength=periodictable.Cu.K_alpha)

# acces isotopes using mass number subscripts
all_isotope = [x.isotope for x in periodictable.Ni]
periodictable.Ni[58].neutron.coherent
periodictable.Ni[62].neutron.coherent

# access ion charge using subscripts
all_ion = periodictable.Ni[58].ion.ionset
periodictable.Ni[58].ion[1].charge
periodictable.Ni[58].ion[-1].charge

all_symbol = [x.symbol for x in periodictable.elements] #119
all_name = [x.name for x in periodictable.elements]

element_number_to_info = {x.number:(x.symbol,x.mass) for x in periodictable.elements}

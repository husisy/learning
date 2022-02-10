import os
import netCDF4

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

file0 = hf_file('test.nc')
root_group = netCDF4.Dataset(file0, 'w', format='NETCDF4')
root_group.data_model #NETCDF4
root_group.close()

z0 = netCDF4.Dataset(file0, "a")
fcstgrp = z0.createGroup("forecasts")
analgrp = z0.createGroup("analyses")
z0.groups

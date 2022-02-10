import os
import numpy as np
import pandas as pd
import xarray as xr

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

## basic00
z0 = xr.DataArray(pd.Series(range(3), index=list('abc'), name='foo'))
z1 = xr.DataArray(np.random.randn(2, 3), dims=('x','y'), coords={"x": [10,20]})
z2 = xr.DataArray(np.random.randn(3), dims='x')
z3 = xr.DataArray(np.random.randn(4), dims='y')

z1.values #np
z1.dims #('x','y')
z1.coords
z1.attrs #store arbitrary metadata

z1.x
z1.coords['x']
z1.x.attrs

z1[0]
z1[0, :]
z1.loc[10]
z1.isel(x=0)
z1.sel(x=10)

z1 + 10
np.sin(z1)
z1.T
z1.sum()
z1.mean(dim='x')

z2 + z3 #(3,4)
z1 - z1.T #(2,3) all zero

z1_labels = xr.DataArray(['E','F','E'], coords=[z1.coords['y']], name='labels')
z1.groupby(z1_labels).mean('y') #(2,2)
z1.groupby(z1_labels).mean(('x','y')) #(2)
z1.groupby(z1_labels).mean() #(2,2)
z1.groupby(z1_labels).map(lambda x: x - x.min()) #min(('x','y'))

z1.attrs["long_name"] = "random velocity"
z1.attrs["units"] = "metres/sec"
z1.attrs["description"] = "A random variable created as an example."
z1.attrs["random_attribute"] = 123
z1.x.attrs["units"] = "x units"
# z1.plot()

z1.to_series().to_xarray() #to pandas and back

## dataset
tmp0 = xr.DataArray(np.random.randn(2, 3), dims=('x','y'), coords={"x": [10,20]})
z0 = xr.Dataset(dict(foo=tmp0, bar=("x", [1, 2]), baz=np.pi))
z0['foo']
z0.foo

netcdf_path = hf_file('example.nc')
z0.to_netcdf(netcdf_path)
z1 = xr.open_dataset(netcdf_path)

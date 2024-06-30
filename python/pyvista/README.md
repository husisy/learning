# pyvista

1. link
   * [github/pyvista](https://github.com/pyvista/pyvista)
   * [pyvista/tutorial](https://tutorial.pyvista.org/getting-started.html)
   * [pyvista/user-guide](https://docs.pyvista.org/version/stable/user-guide/what-is-a-mesh)
   * [pyviz](https://pyviz.org/scivis/index.html)
   * [github/trame](https://github.com/kitware/trame)
   * [github/meshio](https://github.com/nschloe/meshio)
   * [thingiverse](https://www.thingiverse.com/)
   * [pyvista-doc/data-model](https://docs.pyvista.org/version/stable/user-guide/data_model.html)
2. install
   * `mamba install -c conda-forge pyvista jupyterlab trame ipywidgets`
   * `pip install "pyvista[jupyter]"`
3. alternative
   * [github/open3d](https://github.com/isl-org/Open3D)
   * [github/trimesh](https://github.com/mikedh/trimesh)
   * [github/vedo](https://github.com/marcomusy/vedo)
   * [github/polyscope](https://github.com/nmwsharp/polyscope)
   * [github/vispy](https://github.com/vispy/vispy)
   * [github/mayavi](https://github.com/enthought/mayavi)
4. `Jupyter-Server-Proxy` required for remote jupyterlab (see code block below)
5. concept
   * mesh
   * grid
   * volume
   * spatially referenced datasets
   * cell data
   * point data
   * field data
   * eye dome lighting
   * triangulation
6. supported file format `help(pyvista.core.utilities.reader.get_reader)`

```Python
import pyvista
mesh = pyvista.examples.download_lucy()
mesh.plot(smooth_shading=True, color='white')
```

## pyvista jupyterlab on remote server

success on jupyterlab

```bash
pip install jupyter-server-proxy
jupyter server extension enable --sys-prefix jupyter_server_proxy
```

```Python
import pyvista
pyvista.global_theme.trame.server_proxy_enabled = True
# pyvista.global_theme.trame.server_proxy_prefix = '/proxy/' #default, no need to set manually
```

for vscode jupyter

```Python
import pyvista
pyvista.global_theme.trame.server_proxy_enabled = True
pyvista.global_theme.jupyter_backend = "static"
```

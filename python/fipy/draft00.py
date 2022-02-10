import numpy as np
import scipy.special
import matplotlib.pyplot as plt

import fipy

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

# from fipy import Variable, FaceVariable, CellVariable, Grid1D, ExplicitDiffusionTerm, TransientTerm, DiffusionTerm, Viewer
# from fipy.tools import numerix

nx = 50
dx = 1
L = nx*dx
D = 1

mesh = fipy.Grid1D(nx=nx, dx=dx)
phi = fipy.CellVariable(name="solution variable", mesh=mesh, value=0.)
phi.constrain(0, mesh.facesRight)
phi.constrain(1, mesh.facesLeft)

phiAnalytical = fipy.CellVariable(name="analytical value", mesh=mesh)

viewer = fipy.Viewer(vars=(phi, phiAnalytical), datamin=0., datamax=1.)
viewer.plot()

## explicit finite difference
phi.setValue(0) #inital value
timeStepDuration = 0.9 * dx**2 / (2 * D)
steps = 100
phiAnalytical.setValue(1 - scipy.special.erf(mesh.cellCenters.value[0]/(2*np.sqrt(D*timeStepDuration*steps))))
eqX = fipy.TransientTerm() == fipy.ExplicitDiffusionTerm(coeff=D)
for _ in range(steps):
    eqX.solve(var=phi, dt=timeStepDuration)
    viewer.plot()
hfe(phiAnalytical.value, phi.value, eps=1e-3) #around 0.026

## implicit finite difference (can use larger timeStepDuration)
eqI = fipy.TransientTerm() == fipy.DiffusionTerm(coeff=D)
phi.setValue(0) #inital value
timeStepDuration = 10
steps = 10
phiAnalytical.setValue(1 - scipy.special.erf(mesh.cellCenters.value[0]/(2*np.sqrt(D*timeStepDuration*steps))))
for _ in range(steps):
    eqI.solve(var=phi, dt=timeStepDuration)
    viewer.plot()

## Crank-Nicholson scheme
eqCN = eqX + eqI
phi.setValue(0) #inital value
timeStepDuration = 10
steps = 10
phiAnalytical.setValue(1 - scipy.special.erf(mesh.cellCenters.value[0]/(2*np.sqrt(D*timeStepDuration*steps))))
for step in range(steps - 1):
    eqCN.solve(var=phi, dt=timeStepDuration)
    viewer.plot()
eqI.solve(var=phi, dt=timeStepDuration) #one step of the fully implicit scheme to drive down the error
viewer.plot()


## no time evolution
phi.setValue(0)
fipy.DiffusionTerm(coeff=D).solve(var=phi)
viewer.plot()
tmp0 = np.interp(mesh.cellCenters.value[0], np.array([0,nx*dx]), np.array([1,0]))
assert hfe(tmp0, phi.value) < 1e-7

plt.close(viewer.fig)

## boundary condition depends on time
nx = 50
dx = 1
D = 1
time = fipy.Variable()
mesh = fipy.Grid1D(nx=nx, dx=dx)
# del phi.faceConstraints
phi = fipy.CellVariable(name="solution variable", mesh=mesh, value=0.)
phi.constrain(0.5 * (1 + fipy.tools.numerix.sin(time)), mesh.facesLeft)
phi.constrain(0, mesh.facesRight)

viewer = fipy.Viewer(vars=(phi,), datamin=0., datamax=1.)
viewer.plot()
eqI = fipy.TransientTerm() == fipy.DiffusionTerm(coeff=D)
dt = 0.1
while time() < 15:
    time.setValue(time() + dt)
    eqI.solve(var=phi, dt=dt)
    viewer.plot()

## spatially varying diffusion coefficient
nx = 50
dx = 1
L = nx*dx
mesh = fipy.Grid1D(nx=nx, dx=dx)
D = fipy.FaceVariable(mesh=mesh, value=1.0)
X = mesh.faceCenters[0]
D.setValue(0.1, where=(L / 4. <= X) & (X < 3. * L / 4.))
phi = fipy.CellVariable(mesh=mesh)
phi.faceGrad.constrain([1], mesh.facesRight)
phi.constrain(0, mesh.facesLeft)
phi.setValue(0)
fipy.DiffusionTerm(coeff = D).solve(var=phi)
viewer = fipy.Viewer(vars=(phi,), datamin=0., datamax=300)
viewer.plot()
# TODO

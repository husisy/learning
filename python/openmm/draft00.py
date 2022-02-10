import sys
import openmm
import openmm.app
# from openmm.app import *
# from openmm import *
# from openmm.unit import *

# def hf0(x):
#     for y in [openmm,openmm.unit,openmm.app]:
#         if hasattr(y, x):
#             print(y)

# https://github.com/openmm/openmm/blob/master/examples/input.pdb
pdb = openmm.app.PDBFile('input.pdb')
forcefield = openmm.app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=openmm.app.PME,
        nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
integrator = openmm.LangevinMiddleIntegrator(300*openmm.unit.kelvin, 1/openmm.unit.picosecond, 0.004*openmm.unit.picoseconds)
simulation = openmm.app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
simulation.minimizeEnergy()
simulation.reporters.append(openmm.app.PDBReporter('output.pdb', 1000))
simulation.reporters.append(openmm.app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.step(10000)


prmtop = openmm.app.AmberPrmtopFile('input.prmtop')
inpcrd = openmm.app.AmberInpcrdFile('input.inpcrd')
system = prmtop.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
integrator = openmm.LangevinMiddleIntegrator(300*openmm.unit.kelvin, 1/openmm.unit.picosecond, 0.004*openmm.unit.picoseconds)
simulation = openmm.app.Simulation(prmtop.topology, system, integrator)
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
simulation.minimizeEnergy()
simulation.reporters.append(openmm.app.PDBReporter('output.pdb', 1000))
simulation.reporters.append(openmm.app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
simulation.step(10000)


# conda install -c bioconda gramacs
# gro = openmm.app.GromacsGroFile('input.gro')
# top = openmm.app.GromacsTopFile('input.top', periodicBoxVectors=gro.getPeriodicBoxVectors(), includeDir='/usr/local/gromacs/share/gromacs/top')
# system = top.createSystem(nonbondedMethod=openmm.app.PME, nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
# integrator = openmm.LangevinMiddleIntegrator(300*openmm.unit.kelvin, 1/openmm.unit.picosecond, 0.004*openmm.unit.picoseconds)
# simulation = openmm.app.Simulation(top.topology, system, integrator)
# simulation.context.setPositions(gro.positions)
# simulation.minimizeEnergy()
# simulation.reporters.append(openmm.app.PDBReporter('output.pdb', 1000))
# simulation.reporters.append(openmm.app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
# simulation.step(10000)


# psf = openmm.app.CharmmPsfFile('input.psf')
# pdb = openmm.app.PDBFile('input.pdb')
# params = openmm.app.CharmmParameterSet('charmm22.rtf', 'charmm22.prm')
# system = psf.createSystem(params, nonbondedMethod=openmm.app.NoCutoff, nonbondedCutoff=1*openmm.unit.nanometer, constraints=openmm.app.HBonds)
# integrator = openmm.LangevinMiddleIntegrator(300*openmm.unit.kelvin, 1/openmm.unit.picosecond, 0.004*openmm.unit.picoseconds)
# simulation = openmm.app.Simulation(psf.topology, system, integrator)
# simulation.context.setPositions(pdb.positions)
# simulation.minimizeEnergy()
# simulation.reporters.append(openmm.app.PDBReporter('output.pdb', 1000))
# simulation.reporters.append(openmm.app.StateDataReporter(sys.stdout, 1000, step=True, potentialEnergy=True, temperature=True))
# simulation.step(10000)


platform = openmm.Platform.getPlatformByName('CUDA')


pdb = openmm.app.PDBFile('input.pdb')
forcefield = openmm.app.ForceField('amber99sb.xml', 'tip3p.xml')
modeller = openmm.app.Modeller(pdb.topology, pdb.positions)
modeller.addHydrogens(forcefield)
modeller.addSolvent(forcefield, model='tip3p', padding=1*openmm.unit.nanometer)
system = forcefield.createSystem(modeller.topology, nonbondedMethod=openmm.app.PME)
integrator = openmm.VerletIntegrator(0.001*openmm.unit.picoseconds)
simulation = openmm.app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy(maxIterations=100)
positions = simulation.context.getState(getPositions=True).getPositions()
openmm.app.PDBFile.writeFile(simulation.topology, positions, open('output.pdb', 'w'))


class ForceReporter(object):
    def __init__(self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(openmm.unit.kilojoules/openmm.unit.mole/openmm.unit.nanometer)
        for f in forces:
            self._out.write('%g %g %g\n' % (f[0], f[1], f[2]))


import os
for file in os.listdir('structures'):
    pdb = openmm.app.PDBFile(os.path.join('structures', file))
    simulation.context.setPositions(pdb.positions)
    state = simulation.context.getState(getEnergy=True)
    print(file, state.getPotentialEnergy())

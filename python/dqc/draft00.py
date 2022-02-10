import torch
import dqc
import xitorch.optimize


## Hartree-Fock energy
atomzs = torch.tensor([1, 1])
atom_position = torch.tensor([[1,0,0], [-1,0,0]], dtype=torch.float64, requires_grad=True)
mol = dqc.Mol(moldesc=(atomzs, atom_position), basis="3-21G")
qc = dqc.HF(mol).run()
energy = qc.energy()
energy.backward()
force = -atom_position.grad #np.array([[-0.10326285,0,0], [0.10326285,0,0]])


atomzs, atom_position = dqc.parse_moldesc("H -1 0 0; H 1 0 0")
atom_position.requires_grad_()
mol = dqc.Mol(moldesc=(atomzs, atom_position), basis="3-21G")
qc = dqc.HF(mol).run()
energy = qc.energy()
energy.backward()
force = -atom_position.grad

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.atomzs = torch.tensor([1, 1])
        self.atom_position = torch.nn.Parameter(torch.tensor([[1,0,0], [-1,0,0]], dtype=torch.float64))
    def forward(self):
        mol = dqc.Mol(moldesc=(self.atomzs, self.atom_position), basis="3-21G")
        ret = dqc.HF(mol).run().energy()
        return ret
net = MyModel()
optimizer = torch.optim.Adam(net.parameters())
energy_history = []
for _ in range(1000):
    optimizer.zero_grad()
    if net.atom_position.grad is not None:
        net.atom_position.grad.zero_()
    energy = net()
    energy_history.append(energy.item())
    energy.backward()
    optimizer.step()
net.atom_position.detach().numpy()[0,0]*2 #1.388614266529468

## custom XC
class MyLDAX(dqc.xc.CustomXC):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.p = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    @property
    def family(self):
        # 1 for LDA, 2 for GGA, 4 for MGGA
        return 1

    def get_edensityxc(self, densinfo):
        # densinfo has up and down components
        # .value
        # .grad
        # .lapl
        # .kin
        if isinstance(densinfo, dqc.utils.SpinParam):
            # spin-scaling of the exchange energy
            return 0.5 * (self.get_edensityxc(densinfo.u * 2) + self.get_edensityxc(densinfo.d * 2))
        else:
            rho = densinfo.value.abs() + 1e-15  # safeguarding from nan
            return self.a * rho ** self.p

myxc = MyLDAX()
atomzs = torch.tensor([1, 1])
atom_position = torch.tensor([[1,0,0], [-1,0,0]], dtype=torch.float64, requires_grad=True)
mol = dqc.Mol(moldesc=(atomzs, atom_position), basis="3-21G")
qc = dqc.KS(mol, xc=myxc).run()
energy = qc.energy() #-0.4645076142756398
energy.backward()
myxc.a.grad #0.07113085697097354
myxc.p.grad #-0.21081262040462853


# alchemical perturbation
basis = dqc.loadbasis("7:3-21G")

def get_energy(s, lmbda):
    atomzs = 7.0 + torch.tensor([1.0, -1.0], dtype=torch.float64) * lmbda
    atomposs = torch.tensor([[-0.5, 0, 0], [0.5, 0, 0]], dtype=torch.float64) * s
    mol = dqc.Mol((atomzs, atomposs), spin=0, basis=[basis, basis])
    qc = dqc.HF(mol).run()
    return qc.energy()

lmbda = torch.tensor(0, dtype=torch.float64, requires_grad=True)
s0_n2 = torch.tensor(2.04, dtype=torch.float64)  # initial guess of the distance
smin_n2 = xitorch.optimize.minimize(get_energy, s0_n2, params=(lmbda,), method="gd", step=1e-2)
print(smin_n2) #2.045954311218767

grad_lmbda = torch.autograd.grad(smin_n2, lmbda, create_graph=True)[0] #1.7763568394002505e-15
grad2_lmbda = torch.autograd.grad(grad_lmbda, lmbda, create_graph=True)[0] #0.13229581238517224

smin_co = smin_n2 + 0.5*grad2_lmbda #2.112102217411355
smin_bf = smin_n2 + 2*grad2_lmbda #2.310545935989115

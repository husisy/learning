import numpy as np
import pyscf
import pyscf.dft
import pyscf.adc


mol_h2 = pyscf.M(atom='H 0 0 0; H 0 0 1.2', basis='ccpvdz')
mol_h2o = pyscf.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='ccpvdz')
mol_h2o_0 = pyscf.M(atom = 'O 0 0 0; H 0 1 0; H 0 0 1', basis='ccpvdz')
mol_h2o_ccpvtz = pyscf.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587', basis='ccpvtz')
mol_o2 = pyscf.M(atom='O 0 0 0; O 0 0 1.2', spin=2) # (n+2 alpha, n beta) electrons
mol_c2 = pyscf.M(atom = 'C 0 0 .625; C 0 0 -.625', symmetry='d2h')
mol_co = pyscf.M(atom='C 0 0 0.625; O 0 0 -0.625')
mol_HF = pyscf.M(atom='H 0 0 0; F 0 0 1.1', basis='ccpvdz', symmetry=True)

mol = mol_o2
mol.atom
mol.basis
mol.unit
mol.charge
mol.spin
mol.nucmod #nuclear charge model: 0-point charge, 1-Gaussian distribution
mol.natm
mol.atom_coords() #default in Bohr unit
mol.topgroup #detected point group
mol.groupname #supported point group
mol.symmetry
mol.symmetry_subgroup
mol.symm_orb
mol.irrep_name
mol.irrep_id
# mol.mass
mol._atom


## Hartree-Fock
rhf_h2o = pyscf.scf.RHF(mol_h2o)
rhf_h2o.kernel() # -76.0267656731181
pop_and_charge,dip = rhf_h2o.analyze() #Orbital energies, Mulliken population

uhf_o2 = pyscf.scf.UHF(mol_o2)
uhf_o2.kernel() #-147.63345273344964
uhf_o2.spin_square() #(2.003409434468705, 3.0022720959091664)
rohf_o2 = pyscf.scf.ROHF(mol_o2)
rohf_o2.kernel() #-147.63165528656145

rhf_newton_h2o = pyscf.scf.newton(rhf_h2o) #generate initial orbitals if initial guess is not given.
rhf_newton_h2o.kernel() #-76.02676567311926

rhf_h2o_coarse = pyscf.scf.RHF(mol_h2o)
rhf_h2o_coarse.conv_tol = 0.1 #coarse converge, then use newton to finer converge
rhf_h2o_coarse.kernel() #-76.0238705593024
rhf_h2o_fine = pyscf.scf.RHF(mol_h2o).newton()
rhf_h2o_fine.kernel(rhf_h2o_coarse.mo_coeff, rhf_h2o_coarse.mo_occ) #-76.02676567310138

rhf_c2 = pyscf.scf.RHF(mol_c2)
rhf_c2.irrep_nelec = {'Ag': 4, 'B1u': 4, 'B2u': 2, 'B3u': 2}
rhf_c2.kernel() #-74.42166721592592

## Kohn-Sham DFT
uks_h2o = pyscf.dft.UKS(mol_h2o)
uks_h2o.kernel() #-75.85470246168302

uks_pbe_h2o = pyscf.dft.UKS(mol_h2o, xc='pbe,pbe')
uks_pbe_h2o.kernel() #-76.33345765899031

rks_b3lyp_h2o = pyscf.dft.RKS(mol_h2o_ccpvtz, xc='b3lyp')
rks_b3lyp_h2o.kernel() #-76.42271047784816

rks_customXC_h2o = pyscf.dft.RKS(mol_h2o_ccpvtz)
rks_customXC_h2o.xc = '.2 * HF + .08 * LDA + .72 * B88, .81 * LYP + .19 * VWN' # B3LYP
rks_customXC_h2o.grids.atom_grid = (100, 770)
rks_customXC_h2o.grids.prune = None
rks_customXC_h2o.kernel() #-76.4227106527064

rks_vdw_c2 = pyscf.dft.RKS(mol_c2, xc='wb97m_v')
rks_vdw_c2.nlc = 'vv10' #non-local dispersion corrections (van der Waals)
rks_vdw_c2.grids.atom_grid = (99,590)
rks_vdw_c2.grids.prune = None
rks_vdw_c2.nlcgrids.atom_grid = (50,194)
rks_vdw_c2.nlcgrids.prune = pyscf.dft.gen_grid.sg1_prune
rks_vdw_c2.kernel() #-74.91466246669614


# integrals, density fitting
eri_4fold = pyscf.ao2mo.kernel(mol_h2o, rhf_h2o.mo_coeff) #(np,float64,(300,300))
# eri_4fold = mol_h2o.ao2mo(rhf_h2o.mo_coeff)

hcore_ao = mol_h2o.intor_symmetric('int1e_kin') + mol_h2o.intor_symmetric('int1e_nuc')
hcore_mo = np.einsum('pi,pq,qj->ij', rhf_h2o.mo_coeff, hcore_ao, rhf_h2o.mo_coeff)
eri_4fold_ao = mol_h2o.intor('int2e_sph', aosym=4) #(np,float64,(300,300))
eri_4fold_mo = pyscf.ao2mo.incore.full(eri_4fold_ao, rhf_h2o.mo_coeff) #(np,float64,(300,300))


## density fitting techniques
rhf_c2_df = pyscf.df.density_fit(pyscf.scf.RHF(mol_c2), auxbasis='def2-universal-jfit')
# rhf_c2_df = rhf_c2.density_fit(auxbasis='def2-universal-jfit')

## correlated wavefunction theory
mp2_c2 = pyscf.mp.MP2(rhf_c2)
mp2_c2.kernel()[0] #E(MP2) = -74.6665929542934  E_corr = -0.24492573836745
# mp2_c2_df = pyscf.mp.MP2(rhf_c2_df)
# mp2_c2_df.kernel()[0] #fail

ccsd_h2o = pyscf.cc.CCSD(rhf_h2o, frozen=1)
ccsd_h2o.direct = True # AO-direct algorithm to reduce I/O overhead
e_ccsd = ccsd_h2o.kernel()[1] #E(CCSD) = -76.23801446776915  E_corr = -0.2112487946508636
e_ccsd_t = e_ccsd + ccsd_h2o.ccsd_t() #CCSD(T) correction = -0.00303780523256648
# e_ccsd, e_ccsd_t #(np,float64,(4,19))

e_ip_ccsd = ccsd_h2o.ipccsd(nroots=1)[0] #0.4335092441718382
e_ea_ccsd = ccsd_h2o.eaccsd(nroots=1)[0] #0.1673578636248081
e_ee_ccsd = ccsd_h2o.eeccsd(nroots=1)[0] #0.2756370532554109

adc_h2o = pyscf.adc.ADC(rhf_h2o)
e_ip_adc2 = adc_h2o.kernel()[0] # IP-ADC(2) for 1 root
adc_h2o.method = "adc(2)-x"
adc_h2o.method_type = "ea"
e_ea_adc2x = adc_h2o.kernel()[0] # EA-ADC(2)-x for 1 root
adc_h2o.method = "adc(3)"
adc_h2o.method_type = "ea"
e_ea_adc3 = adc_h2o.kernel(nroots = 3)[0] # EA-ADC(3) for 3 roots

# 108,395 MB is required
# fci_h2o = pyscf.fci.FCI(rhf_h2o)
# e_fci = fci_h2o.kernel()[0]


## QM/MM
coords = np.random.random((5, 3)) * 10.
charges = (np.arange(5.) + 1.) * -.1
rhf_h2o_qmmm = pyscf.qmmm.mm_charge(rhf_h2o, coords, charges)
rhf_h2o_qmmm.kernel() #-76.02947189366701
ccsd_h2o_qmmm = pyscf.cc.CCSD(rhf_h2o_qmmm)
e_ccsd = ccsd_h2o_qmmm.kernel()[1] #E(CCSD) = -76.24267979074494  E_corr = -0.2132078970779431

## periodic boundary conditions
tmp0 = '''C     0.      0.      0.
        C     .8917    .8917   .8917
        C     1.7834  1.7834  0.
        C     2.6751  2.6751   .8917
        C     1.7834  0.      1.7834
        C     2.6751   .8917  2.6751
        C     0.      1.7834  1.7834
        C     .8917   2.6751  2.6751'''
cell_diamond = pyscf.pbc.M(atom=tmp0, basis='gth-szv', pseudo='gth-pade', a=np.eye(3)*3.5668)
# a: each row denotes a prmitive vector
# pseudo(optional): crystal pseudo potential
# .ecp: molecular effective core potential

# 5minutes+ (not finished once)
kpts = cell_diamond.make_kpts([4] * 3) # 4 k-poins for each axis
krks_diamond = pyscf.pbc.dft.KRKS(cell_diamond, kpts).density_fit(auxbasis='weigend')
krks_diamond.xc = 'bp86'
krks_diamond = krks_diamond.newton()
krks_diamond.kernel()

rhf_diamond = pyscf.pbc.scf.RHF(cell_diamond).density_fit()
rhf_diamond.kernel()
ccsd_diamond = pyscf.cc.CCSD(rhf_diamond)
ccsd_diamond.kernel()

# CH4 structure optimization
load gs.xml
set wf_dyn PSDA
set xc PBE
set scf_tol 1.e-8
set atoms_dyn CG
run 10 10
save cg.xml

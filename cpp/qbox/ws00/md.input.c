# CH4 molecular dynamics simulation
load cg.xml
set wf_dyn PSDA
set xc PBE
set scf_tol 1.e-6
set dt 10
set atoms_dyn MD
randomize_v 400
run 100 10
save md.xml

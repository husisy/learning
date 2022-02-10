import dpdata

# 'vasp/xml'
dsys = dpdata.LabeledSystem('vasprun.h2o.md.10.xml', fmt='vasp/xml')
dsys.to('deepmd/npy', 'deepmd_data', set_size = dsys.get_nframes())

dsys = dpdata.LabeledSystem('OUTCAR', fmt='vasp/outcar')
dsys.to('deepmd/npy', 'deepmd_data', set_size = dsys.get_nframes())

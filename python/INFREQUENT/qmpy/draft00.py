import qmpy_rester

with qmpy_rester.QMPYRester() as q:
    tmp0 = {'element_set':'(Fe-Mn),O', 'stability': '0.1', 'natom': '<10'}
    # composition include (Fe OR Mn) AND O
    # hull distance smaller than -0.1 eV
    # number of atoms less than 10
    z0 = q.get_oqmd_phases(**tmp0)
z0['links'] #dict next previous base_url meta
z0['resource'] #dict
z0['data']
#(list,dict), name entry_id calculation_id icsd_id formationenergy_id duplicate_entry_id composition
#    composition_generic prototype spacegroup volume ntypes natoms unit_cell sites band_gap delta_e stability fit calculation_label
z0['meta'] #dict query/representation api_version time_stamp data_returned data_available comments query_tree more_data_available
z0['response_message'] #str

with qmpy_rester.QMPYRester() as q:
    z1 = q.get_oqmd_phase_by_id(fe_id=4061139, fields='!sites') # Default: fields=None

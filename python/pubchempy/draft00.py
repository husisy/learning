import pubchempy

z0 = pubchempy.Compound.from_cid(5090)
z0.molecular_formula #C17H14O4S
z0.molecular_weight #314.4
z0.isomeric_smiles #CS(=O)(=O)C1=CC=C(C=C1)C2=C(C(=O)OC2)C3=CC=CC=C3
z0.xlogp #2.3
z0.iupac_name #3-(4-methylsulfonylphenyl)-4-phenyl-2H-furan-5-one
z0.synonyms #[rofecoxib, ...]

z0 = pubchempy.get_compounds('Glucose', 'name') #name smiles sdf inchi inchikey formula

z0 = pubchempy.get_compounds('Aspirin', 'name', record_type='3d')

# z0 = pubchempy.get_compounds('CC', searchtype='superstructure', listkey_count=3) #fail

z0 = pubchempy.get_cids('2-nonenal', 'name', 'substance', list_return='flat') #[17166, 5283335, 5354833]
z0 = pubchempy.get_cids('Glucose', 'name', 'substance', list_return='flat') #[206, 5793, 24749, 64689, 79025, 107526, 5282499]

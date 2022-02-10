import os
import shelve

hf_file = lambda *x: os.path.join('tbd00', *x)
assert os.path.isdir(hf_file())

shelve_name = 'shelve00'

with shelve.open(hf_file(shelve_name)) as db_shelve:
    db_shelve['233'] = 233 #take care mutable object
with shelve.open(hf_file(shelve_name)) as db_shelve:
    tmp0 = db_shelve['233']

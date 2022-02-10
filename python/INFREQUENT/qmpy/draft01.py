import os
import time
import random
import pickle
import traceback
import qmpy_rester
from tqdm import tqdm


def dump_data(new_data=None, db_filepath='oqmd_data.pkl'):
    if not os.path.exists(db_filepath):
        with open(db_filepath, 'wb') as fid:
            pickle.dump({}, fid)
        all_data = {}
    else:
        with open(db_filepath, 'rb') as fid:
            all_data = pickle.load(fid)
    if new_data is not None:
        all_data.update(new_data)
        with open(db_filepath, 'wb') as fid:
            pickle.dump(all_data, fid)
    return all_data


fail_offset = []
num_page = 12753
field_name = ['name', 'entry_id', 'composition', 'volume', 'stability', 'band_gap', 'delta_e']
all_data = dict()
with qmpy_rester.QMPYRester() as q:
    print('please verified num_page={} manually as shown in website http://oqmd.org/oqmdapi/formationenergy'.format(num_page))
    for offset_i in tqdm(range(0, (num_page+100)*50, 50)): #add 100 for redundancy
        try:
            tmp0 = {
                'fields':','.join(field_name),
                'limit': 50,
                'offset': offset_i,
            }
            response = q.get_oqmd_phases(verbose=False, **tmp0)
            assert response['response_message']=='OK'
            all_data.update({x['entry_id']:{y:x[y] for y in field_name} for x in response['data']})
            if response['links']['next'] is None:
                break
            time.sleep(random.uniform(0.1, 0.5))
        except Exception:
            print('offset_i={} failed'.format(offset_i))
            fail_offset.append(offset_i)
            traceback.print_exc()

all_data = dump_data(all_data)
with open('fail_offset.pkl', 'wb') as fid:
    pickle.dump(fail_offset, fail_offset)

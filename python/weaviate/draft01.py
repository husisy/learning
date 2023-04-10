import os
import string
import json
import dotenv
import numpy as np

import weaviate

dotenv.load_dotenv()

np_rng = np.random.default_rng()

def random_vector(dim=4):
    tmp0 = np_rng.normal(size=dim)
    ret = tmp0 / np.linalg.norm(tmp0)
    return ret

def random_integer(min_=0, max_=2):
    assert min_<max_
    ret = int(np_rng.integers(min_, max_)) #[min,max)
    return ret

def random_string(min_=3, max_=10):
    assert min_<max_
    tmp0 = np_rng.integers(min_, max_)
    ind0 = np_rng.integers(0, len(string.ascii_letters), size=tmp0)
    ret = ''.join([string.ascii_letters[x] for x in ind0])
    return ret


tmp0 = weaviate.auth.AuthApiKey(os.environ['WEAVIATE_API_KEY'])
client = weaviate.Client(url=os.environ['WEAVIATE_API_URL'], auth_client_secret=tmp0)

if 'Example' in {x['class'] for x in client.schema.get()['classes']}:
    client.schema.delete_class("Example")
class_schema = {
    "class": "Example",
    "description": "xxx",
    "properties": [
        {"name": "name0", "dataType": ["int"], "description": "yyy"},
        {"name": "name1", "dataType": ["text"], "description": "zzz"},
    ],
    "vectorizer": "none", #text2vec-openai
}
client.schema.create_class(class_schema)


data_list = [{'name0':random_integer(), 'name1':random_string()} for x in range(8)]
uuid_list = []
# https://github.com/openai/openai-cookbook/tree/main/examples/vector_databases/weaviate
# client.batch.configure(batch_size=10,  dynamic=True, timeout_retries=3)
with client.batch as batch:
    batch.batch_size=4
    for x in data_list:
        tmp0 = batch.add_data_object(x, class_name='Example', vector=random_vector())
        uuid_list.append(tmp0)
        # uuid='xxx' #optional, if not provided, one is going to be generated

# if uuid=None (default), return all data objects
z0 = client.data_object.get(uuid=None, limit=41, offset=0, class_name='Example', with_vector=True)
z0['deprecations'] #[]
z0['ojects'] #list of data objects
z0['totalResults'] #8


vector_i = random_vector()
tmp0 = ['name0', 'name1', '_additional {certainty distance vector id}']
z0 = client.query.get('Example', tmp0).with_near_vector({'vector':vector_i}).with_limit(8).do()['data']['Get']['Example']
# z0 = client.query.get('Example', ['name0','name1']).with_near_vector({'vector':vector_i}).with_limit(8).with_additional(['certainty','vector','id']).do()['data']['Get']['Example']
np0 = np.array([np.dot(x0['_additional']['vector'], vector_i) for x0 in z0])
tmp1 = np.array([x0['_additional']['certainty'] for x0 in z0])
tmp2 = np.array([x0['_additional']['distance'] for x0 in z0])
assert np.abs((np0 + 1)/2 - tmp1).max() < 1e-6
assert np.abs((1-np0) - tmp2).max() < 1e-6

tmp0 = {"path": ["name0"], "operator": "Equal", "valueInt": 0}
z0 = client.query.get('Example', ['name0','name1']).with_where(tmp0).do()['data']['Get']['Example']

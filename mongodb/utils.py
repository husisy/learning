import sys
from pymongo import MongoClient


def get_test_client_db_collection(remove_exist=True, collection_name='tbd00'):
    assert sys.version_info >= (3,6) #make sure dictionary is guaranteed to be insertion order
    # see https://docs.python.org/3.7/library/stdtypes.html#mapping-types-dict
    # Changed in version 3.7: Dictionary order is guaranteed to be insertion order. This behavior was an implementation detail of CPython from 3.6
    database_name = 'test'
    login_info = { #local
        'host': 'localhost',
        'port': 23333,
        'username': 'xxx-username',
        'password': 'xxx-password',
        'authSource': database_name,
        'authMechanism': 'SCRAM-SHA-256',
    }
    client = MongoClient(**login_info)
    assert database_name in set(client.list_database_names())
    db = client[database_name]
    if remove_exist and (collection_name in set(db.list_collection_names())):
        _ = db.drop_collection(collection_name)
    collection = db[collection_name]
    print('WARNING: remember to call "client.close()"')
    return client, db, collection


def test_python_version_behavior():
    '''
    already know:
        python 3.5.6: fail
        python 3.6.8: pass
    '''
    print(sys.version)
    database_name = 'test'
    collection_name = 'tbd00'
    login_info = { #local
        'host': 'localhost',
        'port': 23333,
        'username': 'xxx-username',
        'password': 'xxx-password',
        'authSource': database_name,
        'authMechanism': 'SCRAM-SHA-256',
    }
    client = MongoClient(**login_info)
    assert database_name in set(client.list_database_names())
    db = client[database_name]
    if collection_name in set(db.list_collection_names()):
        _ = db.drop_collection(collection_name)
    collection = db[collection_name]
    collection.insert_one({'a':'a', 'b':'b', 'fake_id':'0'})
    collection.insert_one({'b':'b', 'a':'a', 'fake_id':'1'})
    tmp1 = collection.find_one({'fake_id':'0'})
    tmp1.pop('_id')
    print(tmp1, list(tmp1.items()))
    tmp2 = collection.find_one({'fake_id':'1'})
    tmp2.pop('_id')
    print(tmp2, list(tmp2.items()))
    client.close()

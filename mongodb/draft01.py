import time
import datetime
from bson.son import SON

from utils import get_test_client_db_collection

client, db, collection = get_test_client_db_collection()
tmp0 = [
    {'x':1, 'tags':['dog', 'cat']},
    {'x':2, 'tags':['cat']},
    {'x':2, 'tags':['mouse', 'cat', 'dog']},
    {'x':3, 'tags':[]},
]
_ = collection.insert_many(tmp0)
client.close()

# client, db, collection = get_test_client_db_collection()
# pipeline = [
#     {'$unwind': '$tags'},
#     {'$group': {'_id':'$tags', 'count': {'$sum': 1}}},
#     {'$sort': SON([('count', -1), ('_id', -1)])},
# ]
# list(collection.aggregate(pipeline))
# db.command('aggregate', 'tbd00', pipeline=pipeline, explain=True) #TODO

def utc2local (utc):
    # https://stackoverflow.com/a/19238551
    epoch = time.mktime(utc.timetuple())
    offset = datetime.datetime.fromtimestamp(epoch) - datetime.datetime.utcfromtimestamp(epoch)
    return utc + offset

client, db, collection = get_test_client_db_collection()
tmp0 = datetime.datetime.utcnow()
collection.insert_one({'date': tmp0, 'fake_id':0})
tmp1 = list(collection.find({'fake_id':0}, {'date':1,'_id':0}))[0]['date']
client.close()

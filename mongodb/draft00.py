import datetime
import pymongo
from pymongo import MongoClient
from bson.objectid import ObjectId

from utils import get_test_client_db_collection

client, db, collection = get_test_client_db_collection()

post = {
    'author': 'Mike',
    'fake_id': 0,
    'tag': ['mongodb', 'python', 'pymongo'],
    'date': datetime.datetime.utcnow(),
}
post_id = collection.insert_one(post).inserted_id #bson.objectid.ObjectId
post_id_str = str(post_id)
post_id_ = ObjectId(post_id_str)

tmp0 = [
    {'author':'Mike', 'fake_id':1,'tags':['bulk', 'insert'], 'date':datetime.datetime(2009,11,12,11,14)},
    {'author':'Eliot', 'fake_id':2, 'title': 'MongoDB is fun', 'date': datetime.datetime(2009,11,10,10,45)},
]
_ = collection.insert_many(tmp0).inserted_ids

collection.find_one({'author':'Mike'})
collection.find_one({'author': 'laji'}) #None

collection.count_documents({})
collection.count_documents({'author': 'Mike'})
tmp0 = {'date': {'$lt': datetime.datetime(2009,11,12,12)}}
list(collection.find(tmp0).sort('author'))

collection.create_index([('fake_id', pymongo.ASCENDING)], unique=True)
collection.index_information()

client.close()

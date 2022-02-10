# mongodb

1. link
   * [documentation](https://docs.mongodb.com/)
2. community and enterprise
3. document database
4. 优点
   * Documents (i.e. objects) correspond to native data types in many programming language
   * Embedded documents and arrays reduce need for expensive joins
   * Dynamic schema supports fluent polymorphism
5. `_id` field
6. database, collection, document
   * `view`: read only
   * capped collections: fixed-size, high-throughput operations
7. BSON: binary representation of jso
   * [BSON spec](http://bsonspec.org/)
   * data type: `ObjectId`, `str`, `int`, `float`, `array`, `dict`
   * field name `_id` reserved

## mongo shell

1. mongo shell
   * 文件管理器：click `C:/Program Files/MongoDB/Server/4.0/bin/mongo.exe`
   * powershell：`& "C:/Program Files/MongoDB/Server/4.0/bin/mongo.exe"`
   * powershell: `& "C:/Program Files/MongoDB/Server/4.0/bin/mongo.exe" --host localhost --port 27017 --username xxx --authenticationDatabase test`
2. query
   * `$lt`, `$gt`, `$in`, `$or`
   * `/^p/`
3. `quit()`
4. `var x0 = Date();` `typeof x0`
   * `var x1 = new Date();`, `typeof x1`, `x1 instanceof Date`
   * `var x2 = ISODate();`, `typeof x2`

```javascript
help
db.help()
db.collection.hep()
db.collection.find().help()
help misc

show dbs //show databases
show collections

use myNewDB //name case insensitive

db.test00.inserOne({x:233})
db.test01.inserOne({item: "journal", qty: 25, size: {h:14, w:21, uom:"cm"}, tags: ["blank","red"]})
db.test02.insertMany(xxx)
db.createCollection(xxx) //mainly for setting the maximum and documentation validation
```

```javascript
db.inventory.insertMany([
   {item:"journal", qty:25, status:"A", size:{h:14, w:21, uom:"cm"}, tags:["blank", "red"]},
   {item:"notebook", qty:50, status:"A", size:{h:8.5, w:11, uom:"in"}, tags:["red", "blank"]},
   {item:"paper", qty:100, status:"D", size:{ h:8.5, w:11, uom:"in"}, tags:["red", "blank", "plain"]},
   {item:"planner", qty:75, status:"D", size:{h:22.85, w:30, uom:"cm"}, tags:["blank", "red"] },
   {item:"postcard", qty:45, status:"A", size:{h:10, w:15.25, uom:"cm" }, tags:["blue"]}
]);

db.inventory.find({})
db.inventory.find({status:'D'})
db.inventory.find({size: {h:14, w:21, uom:'cm'}}) //order do matter
db.inventory.find({'size.uom': 'cm'})
db.inventory.find({tags:'red'})
db.inventory.find({tags:['red','blank']}) //order do matter
db.inventory.find({status: {$in: ['A','D']}})
db.inventory.find({$or: [{status:'A'},{status:'D'}]})
db.inventory.find({qty: {$lt: 30}})
db.inventory.find({item: /^p/})
```

## MongoDB CRUD

1. Create
   * `db.collection.insertOne()`
   * `db.collection.inserMany()`
   * collection creation
   * `_id` field
   * atomicity
   * write acknowledgement
2. Read
   * `db.collection.find()`
3. Update
   * `db.collection.updateOne()`
   * `db.collection.updateMany()`
   * `db.collection.replaceOne()`
4. Delete
   * `db.collection.deleteOne()`
   * `db.collection.deleteMany()`

## access control

```javascript
use admin
db.createUser(
  {
    user: "root",
    pwd: "root_no_password",
    roles: [ { role: "userAdminAnyDatabase", db: "admin" }, "readWriteAnyDatabase" ]
  }
)
db.createUser(
  {
    user: "xxx-username",
    pwd: "xxx-password",
    roles: [{role: "readWrite", db: "test" }]
  }
)
db.adminCommand({shutdown: 1})
```

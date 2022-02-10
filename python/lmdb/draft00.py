import os
import lmdb

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

lmdb_path = hf_file('tbd00')
env = lmdb.open(lmdb_path, map_size=1099511627776) #1099511627776B=1TB

transaction = env.begin(write=True)
transaction.put(str(1).encode(), "Alice".encode())
transaction.put(str(2).encode(), "Bob".encode())
transaction.delete(str(1).encode())
transaction.commit()

transaction = env.begin() #default write=False
print(transaction.get(str(2).encode()))
for key, value in transaction.cursor():
    print(key, value)

env.close()

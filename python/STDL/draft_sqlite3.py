import os
import sqlite3

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

# example0
db_path = hf_file('example.db')
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)

cmd = '\n'.join([
    'CREATE TABLE stocks',
    '(date text, trans text, symbol text, qty real, price real)'
])
conn.execute(cmd)
conn.execute("INSERT INTO stocks VALUES ('2006-01-05', 'BUY', 'RHAT', 100, 35.14)")
tmp1 = [
    ('2006-03-28', 'BUY', 'IBM', 1000, 45.00),
    ('2006-04-05', 'BUY', 'MSFT', 1000, 72.00),
    ('2006-04-06', 'SELL', 'IBM', 500, 53.00),
]
conn.executemany('INSERT INTO stocks VALUES (?,?,?,?,?)', tmp1)
conn.commit()

print(conn.execute('SELECT * FROM stocks WHERE symbol = ?', ('RHAT',)).fetchone())
for x in conn.execute('SELECT * FROM stocks'):
    print(x)

conn.close()


# example1
db_path = hf_file('example.db')
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('create table user (id varchar(20) primary key, name varchar(20))')
cursor.execute("insert into user (id,name) values ('1', 'Michael')")
cursor.rowcount
cursor.close()
conn.commit()
conn.close()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute('select * from user where id=?', ('1',))
z1 = cursor.fetchall()
cursor.close()
conn.close()

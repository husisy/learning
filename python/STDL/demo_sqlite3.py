import os
import sqlite3

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

def demo_basic():
    database_path = hf_file('example.sqlite')
    if os.path.exists(database_path):
        os.remove(database_path)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cmd = 'CREATE TABLE table00 (key text, ivalue integer, fvalue real)'
    cursor.execute(cmd)

    cursor.execute('INSERT INTO table00 VALUES (?, ?, ?)', ('key00',23,2.3))
    tmp0 = [
        ('key01', 233, 2.33),
        ('key02', 2333, 2.333),
    ]
    cursor.executemany('INSERT INTO table00 VALUES (?, ?, ?)', tmp0)
    cursor.rowcount #2

    print(cursor.execute('SELECT * FROM table00').fetchone())
    print(cursor.execute('SELECT * FROM table00').fetchall())

    cursor.execute('SELECT * FROM table00')
    print(cursor.fetchall())

    cursor.close()
    conn.commit()
    conn.close()

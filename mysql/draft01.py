import pymysql

connection = pymysql.connect(host='localhost', user='xxx-username', password='xxx-password', db='test')

table_name = 'users'
with connection.cursor() as cursor:
    cursor.execute('DROP TABLE IF EXISTS {}'.format(table_name)) #warning when table not exists
    tmp0 = (
        'CREATE TABLE IF NOT EXISTS {} ('
        'id INT NOT NULL AUTO_INCREMENT,'
        'email VARCHAR(255) NOT NULL,'
        'password VARCHAR(255) NOT NULL,'
        'PRIMARY KEY (id)'
        ') ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;'
    ).format(table_name)
    cursor.execute(tmp0)
    connection.commit()

with connection.cursor() as cursor:
    tmp0 = 'INSERT INTO {} (email, password) VALUES (%s, %s)'.format(table_name)
    cursor.execute(tmp0, ('webmaster@python.org', 'very-secret'))
    connection.commit()

with connection.cursor() as cursor:
    cursor.execute('SELECT * FROM users')
    tmp0 = cursor.fetchall()
    print(tmp0)

connection.close()

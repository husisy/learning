import mysql.connector
import contextlib

@contextlib.contextmanager
def mysql_connection_cursor(host='localhost', port=23333, database='test',
        user='xxx-user', password='xxx-password'):
    connection = mysql.connector.connect(host=host, port=port, database=database, user=user, password=password)
    cursor = connection.cursor()
    yield connection,cursor
    cursor.close()
    connection.commit()
    connection.close()

table_name = 'users'
with mysql_connection_cursor() as tmp0:
    connection,cursor = tmp0 #shenmegui pylint warning if put (connection,cursor) in with- expression
    cursor.execute('DROP TABLE IF EXISTS {}'.format(table_name))
    # see https://stackoverflow.com/a/6618385 table_name/field_name cannott be used in cursor.execute(params=)
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

    tmp0 = 'INSERT INTO {} (email, password) VALUES (%s, %s)'.format(table_name)
    cursor.execute(tmp0, ('webmaster@python.org', 'very-secret'))
    connection.commit()

    cursor.execute('SELECT * FROM users')
    tmp0 = cursor.fetchall()
    print(tmp0)

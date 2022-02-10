# SQL 教程

## overview

1. [廖雪峰教程](https://www.liaoxuefeng.com/wiki/001508284671805d39d23243d884b8b99f440bfae87b0f4000)

## miscellaneous

1. NoSQL: MongoDB, Cassandra, Dynamo
2. database model
   * 层次模型
   * 网状模型
   * 关系模型
3. 主流关系数据库
   * 商用数据库：[Oracle](https://www.oracle.com/)，[SQL Server](https://www.microsoft.com/sql-server/)，[DB2](https://www.ibm.com/db2/)
   * 开源数据库：[MySQL](https://www.mysql.com/)，[PostgreSQL](https://www.postgresql.org/)
   * 桌面数据库，以微软[Access](https://products.office.com/access)为代表，适合桌面应用程序使用
   * 嵌入式数据库，以[Sqlite](https://sqlite.org/)为代表，适合手机应用和桌面程序
4. 数据类型: `INT`, `BIGINT`, `REAL`, `DOUBLE`, `DECIMAL(M,N)`, `CHAR(N)`, `VARCHAR(N)`, `BOOLEAN`, `DATE`, `TIME`, `DATETIME`
5. Structured Query Language (SQL)
   * Data Definition Language (DDL): 允许用户定义数据，也就是创建表、删除表、修改表结构这些操作。通常，DDL由数据库管理员执行
   * Data Manipulation Language (DML)：为用户提供添加、删除、更新数据的能力，这些是应用程序对数据库的日常操作
   * Data Query Language (DQL)：DML为用户提供添加、删除、更新数据的能力，这些是应用程序对数据库的日常操作
6. 域domain
7. 记录Record
8. 字段Column
9. 主键与外键，联合主键
   * 不使用任何业务相关的字段作为主键，因此，身份证号、手机号、邮箱地址这些看上去可以唯一的字段，均**不可**用作主键
   * 自增整数类型```BIGINT NOT NULL AUTO_INCREMENT```
   * 全局唯一GUID类型
10. 关系模型
    * 一对一
    * 一对多
    * 多对一
    * 多对多

## MySql

1. mysql
   * `mysqlsh xxx-username@localhost:23333/test`, `/js`, `/sql`, `/quit`, `QUIT;`（所有backslash改为slash）
   * `mysqlsh xxx-username@localhost:23333/test -f init_data.sql`，[廖雪峰demo scripts](https://github.com/michaelliao/learn-sql/blob/master/mysql/init-test-data.sql)
2. 系统数据库：`information_schema`，`mysql`，`performance_schema`，`sys`
3. 创建、删除database是在`sql prompt`中进行
   * `CREATE DATABASE test;`, `CREATE DATABASE IF NOT EXISTS test;`
   * `DROP DATABASE test;`
   * `SHOW DATABASES;`, `USE test`
4. 注释：`#`
5. Python-driver
   * `conda install -n python_cpu -c conda-forge mysql-connector-python`
   * `conda install -n python_cpu -c anaconda pymysql`
6. Object-Relational Mapping (ORM)
   * `conda install -n python_cpu -c conda-forge sqlalchemy`

用户管理权限

```sql
CREATE USER 'xxx'@'%' IDENTIFIED BY 'xxx-password';
GRANT ALL ON *.* TO 'xxx'@'%';
FLUSH PRIVILEGES;
```

操作database

```sql
SHOW DATABASES;
CREATE DATABASE IF NOT EXISTS test;
DROP DATABASE IF EXISTS test;
USE test;
EXIT;
```

操作table

```sql
DROP TABLE IF EXISTS students;
CREATE TABLE students (
    id BIGINT NOT NULL AUTO_INCREMENT,
    class_id BIGINT NOT NULL,
    name VARCHAR(100) NOT NULL,
    gender VARCHAR(1) NOT NULL,
    score INT NOT NULL,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
SHOW TABLES;
DESC students;
SHOW CREATE TABLE students;
ALTER TABLE students ADD COLUMN birth VARCHAR(10) NOT NULL;
ALTER TABLE students CHANGE COLUMN birth birthday VARCHAR(20) NOT NULL;
ALTER TABLE students DROP COLUMN birthday;

```

增删改查（先执行`init_data.sql`创建初始数据）

```sql
USE test;

SELECT 1;
SELECT 100+200;
SELECT version();
SELECT current_date;

SELECT * FROM students;
SELECT * FROM students WHERE score >= 80;
SELECT * FROM students WHERE score >=80 AND gender = 'M';
SELECT * FROM students WHERE score >=80 OR gender = 'M';
SELECT * FROM students WHERE score >=80 OR NOT gender = 'F';
SELECT * FROM students WHERE (score < 80 OR score > 90) AND gender = 'M';
SELECT id, score, name FROM students;
SELECT id, score points, name FROM students;
SELECT id, score points, name FROM students ORDER BY score;
SELECT id, score points, name FROM students ORDER BY points DESC;
SELECT id, name, gender, score FROM students ORDER BY score DESC, gender;
SELECT * FROM students ORDER BY score LIMIT 3 OFFSET 0;

SELECT COUNT(*) FROM students;
SELECT class_id, COUNT(*) num FROM students GROUP BY class_id;
SELECT class_id, gender, COUNT(*) num FROM students GROUP BY class_id, gender;
SELECT class_id, AVG(score) FROM students GROUP BY class_id;

SELECT * FROM students, classes;
SELECT s.id sid, s.name, s.gender, s.score, c.id cid, c.name cname FROM students s, classes c;

SELECT s.id, s.name, s.class_id, c.name cname, s.gender, s.score FROM students s INNER JOIN classes c ON s.class_id=c.id;

INSERT INTO students (class_id, name, gender, score) VALUES (2, 'Da Niu', 'M', 80);
INSERT INTO students (class_id, name, gender, score) VALUES (1, 'Da Bao', 'M', 87), (2, 'Er Bao', 'M', 81);

UPDATE students SET name='Da Niu', score=66 WHERE id=1;
UPDATE students SET name='Xiao Niu', score=77 WHERE id>=5 AND id<=7;
UPDATE students SET score=score+10 WHERE score<80

DELETE FROM students WHERE id=1;

CREATE TABLE students_of_class1 SELECT * FROM students WHERE class_id=1;

CREATE TABLE students_of_class1 SELECT * FROM students WHERE class_id=1;

CREATE TABLE statistics (
    id BIGINT NOT NULL AUTO_INCREMENT,
    class_id BIGINT NOT NULL,
    average DOUBLE NOT NULL,
    PRIMARY KEY (id)
);
INSERT INTO statistics (class_id, average) SELECT class_id, AVG(score) FROM students GROUP BY class_id;

REPLACE INTO students (id, class_id, name, gender, score) VALUES (1, 1, 'Xiao Ming', 'F', 99);
INSERT INTO students (id, class_id, name, gender, score) VALUES (1, 1, '小明', 'F', 99) DUPLICATE KEY UPDATE name='小明', gender='F', score=99;
INSERT IGNORE INTO students (id, class_id, name, gender, score) VALUES (1, 1, '小明', 'F', 99);

BEGIN;
UPDATE students SET score=score-5 WHERE id=1;
UPDATE students SET score=score+5 WHERE id=2;
COMMIT;

BEGIN;
UPDATE students SET score=score+5 WHERE id=1;
UPDATE students SET score=score-5 WHERE id=2;
ROLLBACK;
```

## PostgreSQL

1. PostgreSQL创建、删除database是在cmd/bash中进行

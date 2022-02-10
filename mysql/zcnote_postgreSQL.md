# PostgreSQL tutorial

## overview

1. reference
   * [official site](https://www.postgresql.org/)
   * [PostgreSQL wiki](https://wiki.postgresql.org/wiki/Main_Page): include FAQ, TODO, etc.
2. Object-Relational Database Management System (ORDBMS)
3. features
   * standard SQL
   * complex queries
   * foreign keys
   * triggers
   * updatable view
   * transactional integrity
   * multiversion concurrency control
   * customs features: data types, functions, operators, aggregate functions, index methods, procedural languages
4. environment variable
   * `PATH`: `C:\Program Files\PostgreSQL\11\bin` (win)
   * `PGHOST`
   * `PGPORT`
   * `PATH`
5. create, drop, and connect database
   * `createdb -U postgres test`
   * `dropdb -U postgres test`
   * `psql -h 127.0.0.1 -U postgres -d test -p 5432`
6. postgresql commands
   * `\h`: help on SQL commands
   * `\?`: help on postgresql internal commands
   * `\q`: quit `psql` prompt
   * `\i`: commands from files
7. misc syntax
   * quit: `QUIT;`
   * comment: `--`

```sql
DROP TABLE IF EXISTS classes;
DROP TABLE IF EXISTS students;
CREATE TABLE classes (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);
CREATE TABLE students (
    id BIGSERIAL PRIMARY KEY,
    class_id BIGINT NOT NULL,
    name VARCHAR(100) NOT NULL,
    gender VARCHAR(1) NOT NULL,
    score INT NOT NULL
);

ALTER TABLE classes ADD COLUMN tbd VARCHAR(20) NOT NULL;
ALTER TABLE classes RENAME COLUMN tbd TO tbd01;
ALTER TABLE classes ALTER COLUMN tbd01 TYPE VARCHAR(40);
ALTER TABLE classes DROP COLUMN tbd01;

INSERT INTO classes (id, name) VALUES (1, 'class1');
INSERT INTO classes (id, name) VALUES (2, 'class2');
INSERT INTO classes (id, name) VALUES (3, 'class3');
INSERT INTO classes (id, name) VALUES (4, 'class4');
INSERT INTO students (id, class_id, name, gender, score) VALUES (1, 1, 'Xiao Ming', 'M', 90);
INSERT INTO students (id, class_id, name, gender, score) VALUES (2, 1, 'Xiao Hong', 'F', 95);
INSERT INTO students (id, class_id, name, gender, score) VALUES (3, 1, 'Xiao Jun', 'M', 88);
INSERT INTO students (id, class_id, name, gender, score) VALUES (4, 1, 'Xiao Mi', 'F', 73);
INSERT INTO students (id, class_id, name, gender, score) VALUES (5, 2, 'Xiao Bai', 'F', 81);
INSERT INTO students (id, class_id, name, gender, score) VALUES (6, 2, 'Xiao Bin', 'M', 55);
INSERT INTO students (id, class_id, name, gender, score) VALUES (7, 2, 'Xiao Lin', 'M', 85);
INSERT INTO students (id, class_id, name, gender, score) VALUES (8, 3, 'Xiao Xin', 'F', 91);
INSERT INTO students (id, class_id, name, gender, score) VALUES (9, 3, 'Xiao Wang', 'M', 89);
INSERT INTO students (id, class_id, name, gender, score) VALUES (10, 3, 'Xiao Li', 'F', 85);
INSERT INTO students (class_id, name, gender, score) VALUES (2, 'Da Niu', 'M', 80);
SELECT setval('classes_id_seq', max(id)) FROM classes; --necessary for postgresql-SERIAL
SELECT setval('students_id_seq', max(id)) FROM students;

SELECT 1;
SELECT version();
SELECT current_date;

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

SELECT s.name sname, c.name cname, s.gender, s.score FROM students s INNER JOIN classes c ON s.class_id=c.id;

INSERT INTO students (class_id, name, gender, score) VALUES (2, 'Da Niu', 'M', 80);
INSERT INTO students (class_id, name, gender, score) VALUES (1, 'Da Bao', 'M', 87), (2, 'Er Bao', 'M', 81);

UPDATE students SET name='Da Niu', score=66 WHERE id=1;
UPDATE students SET name='Xiao Niu', score=77 WHERE id>=5 AND id<=7;
UPDATE students SET score=score+10 WHERE score<80

DELETE FROM students WHERE id=1;
```

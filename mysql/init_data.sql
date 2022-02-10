CREATE DATABASE IF NOT EXISTS test;

USE test;

DROP TABLE IF EXISTS classes;
DROP TABLE IF EXISTS students;

CREATE TABLE classes (
    id BIGINT NOT NULL AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE students (
    id BIGINT NOT NULL AUTO_INCREMENT,
    class_id BIGINT NOT NULL,
    name VARCHAR(100) NOT NULL,
    gender VARCHAR(1) NOT NULL,
    score INT NOT NULL,
    PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO classes(id, name) VALUES (1, 'class1');
INSERT INTO classes(id, name) VALUES (2, 'class2');
INSERT INTO classes(id, name) VALUES (3, 'class3');
INSERT INTO classes(id, name) VALUES (4, 'class4');

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

SELECT 'ok' as 'result:';
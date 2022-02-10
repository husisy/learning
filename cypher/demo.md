# cypher demo

## MATCH

see [link-3.3.1](https://neo4j.com/docs/developer-manual/current/cypher/clauses/match/)

```bash
CREATE (x1:Person {name:'Oliver Stone'}), (x2:Person {name:'Michael Douglas'}),
    (x3:Person {name:'Charlie Sheen'}), (x4:Person {name:'Martin Sheen'}), (x5:Person {name:'Rob Reiner'}),
    (y1:Movie {title:'Wall Street'}), (y2:Movie {title:'The American President'})
CREATE (x1)-[:DIRECTED]->(y1), (x5)-[:DIRECTED]->(y2),
    (x2)-[:ACTED_IN {role:'Gordon Gekko'}]->(y1), (x2)-[:ACTED_IN {role:'President Andrew Shepherd'}]->(y2),
    (x3)-[:ACTED_IN {role:'Bud Fox'}]->(y1),
    (x4)-[:ACTED_IN {role:'Carl Fox'}]->(y1), (x4)-[:ACTED_IN {role:'A.J. Maclnerney'}]->(y2)
```

```bash
MATCH (n) RETURN n

MATCH (x:Person {name:'Oliver Stone'}) -[r]-> (y) RETURN type(r)
MATCH p=(:Person) -[:ACTED_IN|:DIRECTED]-> (:Movie) RETURN p
MATCH (x:Person) -[y:ACTED_IN|:DIRECTED]-> (z:Movie) RETURN x.name, type(y), z.title
MATCH (:Person {name:'Charlie Sheen'}) -[:ACTED_IN]-> () <-[:DIRECTED]- (x) RETURN x.name

MATCH p=(:Movie {title:'Wall Street'}) -[*0..1]- (x) RETURN p

MATCH (x:Person {name:'Martin Sheen'}),(y:Person {name:'Oliver Stone'}), p = shortestPath((x)-[*..15]-(y)) RETURN p

MATCH (x:Person {name:'Martin Sheen'}),(y:Person {name:'Oliver Stone'}) RETURN shortestPath((x)-[*..15]-(y))

MATCH (x:Person {name:'Martin Sheen'}),(y:Person {name:'Oliver Stone'})
MATCH p = shortestPath((x)-[*..15]-(y))
RETURN p
```

## RETURN

see [link-3.3.4](https://neo4j.com/docs/developer-manual/current/cypher/clauses/return/)

```bash
CREATE (x:Node {name:'A', happy:TRUE, age:55}), (y:Node {name:'B'})
CREATE (x)-[:BLOCK {name:'BLOCK'}]->(y), (x)-[:KNOW {name:'KNOW'}]->(y)
```

```bash
MATCH p=(x:Node {name:'A'}) -[r]-> (y) RETURN *
MATCH (x:Node {name:'A'}) RETURN x, (x)-->()
MATCH (x:Node {name:'A'})-->(y) RETURN DISTINCT y.name
```

## WITH

see [link-3.3.5](https://neo4j.com/docs/developer-manual/current/cypher/clauses/with/)

```bash
CREATE (x1:Person {name:'Anders'}), (x2:Person {name:'Bossman'}), (x3:Person {name:'Ceasar'}),
    (x4:Person {name:'George'}), (x5:Person {name:'David'})
CREATE (x1)-[:KNOW]->(x2), (x1)-[:BLOCK]->(x3),
    (x2)-[:KNOW]->(x4), (x2)-[:BLOCK]->(x5),
    (x3)-[:KNOW]->(x4),
    (x5)-[:KNOW]->(x1)
```

```bash
MATCH (x:Person {name:'David'})--(y)-->()
WITH y, count(*) AS num1 WHERE num1>1
RETURN y.name
```

```bash
MATCH (x) RETURN x.name ORDER BY x.name DESC LIMIT 3
```

```bash
MATCH (x) RETURN collect(x.name)

MATCH (x)
WITH collect(x.name) AS y
UNWIND y AS z
RETURN collect(z)
```

```bash
MATCH (x:Person {name:'Anders'})--(y)
WITH y ORDER BY y.name DESC LIMIT 1
MATCH (y)--(z) RETURN z.name
```

## ORDER BY

see [link-3.3.8](https://neo4j.com/docs/developer-manual/current/cypher/clauses/order-by/)

```bash
CREATE (A:Node {name:'A', age:34, height:170}),
    (B:Node {name:'B', age:34}),
    (C:Node {name:'C', age:32, height:185})
CREATE (A) -[:KNOWS]-> (B), (B) -[:KNOWS]-> (C)
```

```bash
MATCH (n) RETURN n.name ORDER BY n.name DESC
MATCH (n) RETURN n.name ORDER BY n.height
```

see [link-3.3.9](https://neo4j.com/docs/developer-manual/current/cypher/clauses/skip/)

```bash
CREATE (A:Node {name:'A'}), (B:Node {name:'B'}),
    (C:Node {name:'C'}), (D:Node {name:'D'}), (E:Node {name:'E'})
```

```bash
MATCH (n) RETURN n.name ORDER BY n.name SKIP 1 LIMIT 2
```

## DELETE

see [link-3.3.12](https://neo4j.com/docs/developer-manual/current/cypher/clauses/delete/)

```bash
CREATE (n0:Person {name:'Andy', age:36}), (n1:Person {name:'UNKNOWN'}),
    (n2:Person {name:'Timothy', age:25}), (n3:Person {name:'Peter', age:34})
CREATE (n0)-[:KNOWS]->(n2), (n0)-[:KNOWS]->(n3)
```

```bash
MATCH (x:Person {name:'UNKNOWN'}) DELETE (x)
MATCH (n) DETACH DELETE n
MATCH (x:Person {name:'Timothy'}) DETACH DELETE x
MATCH (:Person {name:'Andy'}) -[r:KNOWS]->() DELETE r
```

## SET

see [link-3.3.13](https://neo4j.com/docs/developer-manual/current/cypher/clauses/set/)

```bash
CREATE (n0:Node {name:'n0'}), (n1:Node {name:'n1'})
```

```bash
MATCH (x:Node) WITH x WHERE x.name='n1' SET x.name='233'
MATCH (x:Node) SET (CASE WHEN x.name='233' THEN x END).name='n1'

MATCH (x:Node {name:'n1'}) SET x.name=NULL
MATCH (x:Node) WHERE NOT exists(x.name) SET x.name='n1'

MATCH (x:Node {name:'n0'}), (y:Node {name:'n1'}) SET x=y
MATCH (x:Node) WITH x LIMIT 1 SET x={name:'n0'}

MATCH (x:Node {name:'n0'}) SET x={}
MATCH (x:Node) WHERE x.name IS NULL SET x+={name:'n0'}

MATCH (x:Node) SET x:Node:Mode
```

## MERGE

see [link-3.3.16](https://neo4j.com/docs/developer-manual/current/cypher/clauses/merge/)

```bash
CREATE (x1:Person {name:'Oliver Stone', bornIn:'New York', chauffeurName:'Bill White'}),
    (x2:Person {name:'Michael Douglas', bornIn:'New Jersey', chauffeurName:'John Brown'}),
    (x3:Person {name:'Charlie Sheen', bornIn:'New York', chauffeurName:'John Brown'}),
    (x4:Person {name:'Martin Sheen', bornIn:'Ohio', chauffeurName:'Bob Brown'}),
    (x5:Person {name:'Rob Reiner', bornIn:'New York', chauffeurName:'Ted Green'}),
    (y1:Movie {title:'Wall Street'}),
    (y2:Movie {title:'The American President'})
CREATE (x1)-[:ACTED_IN]->(y1),
    (x2)-[:ACTED_IN]->(y1), (x2)-[:ACTED_IN]->(y2),
    (x3)-[:ACTED_IN]->(y1), (x3)-[:FATHER]->(x4),
    (x4)-[:ACTED_IN]->(y1), (x4)-[:ACTED_IN]->(y2),
    (x5)-[:ACTED_IN]->(y2)
```

```bash
MATCH (x:Person)
MERGE (y:City {name:x.bornIn})
MERGE (x) -[:BORNIN]-> (y)
MERGE (z:Person {name:x.chauffeurName})
MERGE (x) -[:HAS_CHAUFFEUR]-> (z)
RETURN x,y,z
```

## UNION

see [link-3.3.19](https://neo4j.com/docs/developer-manual/current/cypher/clauses/union/)

```bash
CREATE (x1:Actor {name:'Hitchcock'}),
    (x2:Actor {name:'Anthony Hopkins'}),
    (x3:Actor {name:'Helen Mirren'}),
    (y1:Movie {name:'Hitchcock'})
CREATE (x2)-[:ACTS_IN]->(y1), (x2)-[:KNOWS]->(x3), (x3)-[:ACTS_IN]->(y1)
```

```bash
MATCH (x:Actor) RETURN x.name AS name UNION MATCH (y:Movie) RETURN y.name AS name
MATCH (x:Actor) RETURN x.name AS name UNION ALL MATCH (y:Movie) RETURN y.name AS name
```

## LOAD CSV

see [link-3.3.20](https://neo4j.com/docs/developer-manual/current/cypher/clauses/load-csv/)

1. setting: `dbms.directories.import=/path/to/project`
2. data source
   * `https://neo4j.com/docs/developer-manual/3.4/csv/artists.csv`
   * `https://neo4j.com/docs/developer-manual/3.4/csv/import/persons.csv`
   * `https://neo4j.com/docs/developer-manual/3.4/csv/import/movies.csv`

```bash
LOAD CSV FROM 'file:///artists.csv' AS line
CREATE (:Artist {name: line[1], year:toInteger(line[2])})

LOAD CSV FROM 'file:///artists.csv' AS line FIELDTERMINATOR ','

USING PERIODIC COMMIT 500 LOAD CSV FROM 'file:///artists.csv' AS line

LOAD CSV WITH HEADERS FROM 'file:///artists_with_headers.csv' AS line
CREATE (:Artist {name: line.Name, year:toInteger(line.Year)})
```

```bash
LOAD CSV WITH HEADERS FROM 'file:///persons.csv' AS line
CREATE (:Person {id:toInteger(line.id), name:line.name})

CREATE INDEX ON :Country(name)
CREATE CONSTRAINT ON (x:Person) ASSERT x.id IS UNIQUE
CREATE CONSTRAINT ON (x:Movie) ASSERT x.id IS UNIQUE

LOAD CSV WITH HEADERS FROM 'file:///movies.csv' AS line
MERGE (c:Country {name:line.country})
CREATE (m:Movie {id:toInteger(line.id), title:line.title, year:toInteger(line.year)})
CREATE (m) -[:MADE_IN]-> (c)

USING PERIODIC COMMIT 500
LOAD CSV WITH HEADERS FROM 'file:///roles.csv' as line
MATCH (p:Person {id:toInteger(line.personId)}), (m:Movie {id:toInteger(line.movieId)})
CREATE (p) -[:PLAYED{role:line.role}]-> (m)

DROP INDEX ON :Country(name)
DROP CONSTRAINT ON (x:Person) ASSERT x.id IS UNIQUE
DROP CONSTRAINT ON (x:Movie) ASSERT x.id IS UNIQUE

MATCH (n:Person) REMOVE n.id
MATCH (n:Movie) REMOVE n.id
```

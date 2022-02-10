# cypher

1. reference
   * [documentation](https://neo4j.com/developer/get-started/)
   * [python-driver](https://github.com/neo4j/neo4j-python-driver) `pip install neo4j`

```bash
:server status
:play start
:play write-code
:play movie-graph
```

## Graph database

1. node
   * property
   * label
2. relationship
   * relationship type
   * property
   * outgoing and incoming relationship
   * relationship with self
3. property: named value
   * key: `string`
   * value: `integer`, `float`, `string`, `boolean`, `spatial type(point)`, `temporal type(date, time, localtime, duration)`
   * `null` is not a valid property value, since it can be modeled by the absence of a property key
4. label: assign roles or types to nodes
   * a node can be labeled with any number of labels
5. traversal: navigate through a graph to find paths
6. paths: retrieved from a cypher query or traversal
   * length could be zero, one
7. schema
   * optional
   * master machine only
8. index
9. constraint

## Example

### Atom operation

1. delete: `MATCH (n) DETACH DELETE n`
2. view: `MATCH (n) RETURN n`
3. create: `CREATE (:Node:Mode) -[:R]-> (:Mode:Node)`
4. set: `SET n0.name = 'n0'`
5. remove: `REMOVE n0.name`
6. type: `RETURN Type(r0)`
7. comment: `//comment`

### MERGE

```bash
CREATE (:Node {name:'n0'})
MERGE (n0:Node {name:'n0'}) ON CREATE SET n0.property='2333'
```

### MATCH

```bash
CREATE (:Node {prop:1}), (:Node {prop:2}), (:Node {prop:3})
MATCH (n:Node) WHERE n.prop=2 RETURN n
MATCH (n:Node) WHERE n.prop<>2 RETURN n
MATCH (n:Node) WHERE NOT n.prop=2 RETURN n
MATCH (n:Node) WHERE n.prop>2 RETURN n
MATCH (n:Node) WHERE n.prop<2 RETURN n
MATCH (n:Node) WHERE n.prop<2 OR n.prop>2 RETURN n
MATCH (n:Node) WHERE n.prop>1 AND (NOT n.prop=3) RETURN n
MATCH (n) WHERE n:Node RETURN n
```

```bash
CREATE (:Node {name:'n0'}), (:Node {name:'n1'}), (:Node {name:'n2'})
MATCH (n:Node) WHERE n.name=~'.1' RETURN n
MATCH (n:Node) WHERE n.name=~'.{2}' RETURN n
MATCH (n:Node) WHERE n.name STARTS WITH 'n' RETURN n
```

```bash
CREATE (:Node {p:[233]}), (:Node)
MATCH (n:Node) WHERE n.p IS NOT NULL RETURN n
MATCH (n:Node) WHERE 233 IN n.p RETURN n
```

```bash
CREATE (:Node) -[:R]-> (:Mode)
MATCH (n) WHERE 'Node' IN labels(n) OR 'Mode' IN labels(n) RETURN n
```

```bash
CREATE (:Node {name:'n0'}) -[:R]-> (:Node {name:'n1'})
MATCH (x),(y) WHERE (x)-->(y) RETURN x,y
```

### RETURN

```bash
CREATE (:Node {id:1, p:233, q:'bcc'}), (:Node {id:2, p:2333, q:'bccc'}), (:Node {id:1, q:'bcccc'})
MATCH (p:Node) RETURN p, p.id as id, toUpper(p.q), coalesce(p.p, 'n/a') as nidawoya
```

```bash
CREATE (:Node {name:'n0'}), (:Node {name:'n1'}), (:Node {name:'n2'})
MATCH (n:Node) RETURN count(n) as node
```

```bash
CREATE (:Node {id:1, p:233}), (:Node {id:2, p:332}), (:Node {id:3, p:233})
MATCH (p:Node) RETURN DISTINCT p.p, count(p.p)
```

```bash
CREATE (n1:Node {id:1}), (n2:Node {id:2}), (n3:Node {id:3}), (n4:Node {id:4})
CREATE (m1:Mode {id:1}), (m2:Mode {id:2})
CREATE (n1) -[:Rel {id:1}]-> (m1),
    (n2) -[:Rel {id:2}]-> (m1),
    (n3) -[:Rel {id:3}]-> (m1)
CREATE (n3) -[:Rel {id:4}]-> (m2),
    (n4) -[:Rel {id:5}]-> (m2)

MATCH (n:Node) -[:Rel]-> (m:Mode) RETURN m.id, collect(n.id), count(n.id)

MATCH (n:Node) RETURN n.id AS id, head(labels(n)) AS labels
UNION
MATCH (m:Mode) RETURN m.id AS id, head(labels(m)) AS labels

MATCH (n:Node) -[r]-> (m:Mode)
with n, count(n) AS x1, collect(m) as x2 WHERE x1>1 RETURN n, x1, x2
```

### SCHEMA

```bash
:schema
CREATE INDEX ON :Node(id)
DROP INDEX ON :Node(id)
CREATE CONSTRAINT ON (x:Node) ASSERT x.id IS UNIQUE
DROP CONSTRAINT ON (x:Node) ASSERT x.id IS UNIQUE
```

### WITH COUNT

```bash
CREATE (n0:Node {name:'n0'}), (n1:Node {name:'n1'}), (n2:Node {name:'n2'}), (n3:Node {name:'n3'})
CREATE (n0) -[:R]-> (n1), (n0) -[:R]-> (n2), (n0) -[:R]-> (n3),
    (n1) -[:R]-> (n2), (n1) -[:R]-> (n3), (n2) -[:R]-> (n3)

MATCH (x:Node) -[]-> (y:Node)
WITH x, count(y) AS nodeCount WHERE nodeCount>1
RETURN x, nodeCount

MATCH (x:Node) -[]-> (y)
WITH x, count(y) AS outNodeCount SET x.outNodeCount=outNodeCount
RETURN x, outNodeCount

MATCH (x:Node) <-[]- (y)
WITH x, count(y) AS inNodeCount SET x.inNodeCount=inNodeCount
RETURN x, inNodeCount

MATCH (x:Node) -[]- (y)
WITH x, count(y) AS nodeCount SET x.nodeCount=nodeCount
RETURN x, nodeCount
```

### chain MATCH

```bash
CREATE (n0:Node {name:'n0'}), (n1:Node {name:'n1'}), (n2:Node {name:'n2'})
CREATE (n0) -[:R]-> (n1), (n1) -[:R]-> (n2), (n2) -[:R]-> (n0)

MATCH (x:Node {name:'n0'}) -[:R]- (y) -[:R] - (z) RETURN z

MATCH (x:Node {name:'n0'}) -[:R]- (y), (y) -[:R] - (z) RETURN z

MATCH (x:Node {name:'n0'}) -[:R]- (y)
MATCH (y) -[:R]- (z) RETURN z
```

### CASE

```bash
CREATE (n0:Node {name:'n0'}), (n1:Node {name:'n1'}), (n2:Node {name:'n2'})
CREATE (n0) -[:R]-> (n1), (n1) -[:R]-> (n2), (n2) -[:R]-> (n0)

MATCH (x:Node)
WITH x, CASE x.name WHEN 'n0' THEN '0n' ELSE '233' END AS tmp1
RETURN x, tmp1

MATCH (x:Node)
WITH x, CASE WHEN x.name='n0' THEN '0n' WHEN x.name='n1' THEN '1n' ELSE '233' END AS tmp1
RETURN x, tmp1
```

### DISTINCT COUNT

```bash
CREATE (x1:Node {p:1}), (x2:Node {p:1}), (x3:Node {p:2}), (x4:Node {p:3})
WITH [x1,x2,x3,x4] AS xs
UNWIND xs AS x
RETURN DISTINCT x.p, count(*)
```

### property

```bash
CREATE (:Mode {name: 'aa'}), (:Mode {name: 'bb'})
CREATE (:Node {name:'n1', xx_aa:1, xx_bb:2}), (:Node {name:'n2', xx_aa:3, xx_bb:4})

MATCH (n:Node), (m:Mode)
WHERE n['xx_'+m.name]>2
RETURN DISTINCT n.name
```

### simple expression

```bash
RETURN 233
RETURN 2+3, 2-3, 2*3, 2/3, 5%3, 2^3
RETURN 2.0+3, 2.0-3, 2.0*3, 2.0/3, 5.0%3, 2.0^3
WITH 2 AS x, 3 AS y RETURN x+y
```

```bash
RETURN TRUE, FALSE, 3>2, 3<2, 233=233, 2<>3, 2>=2, 2<=2
RETURN 2<3<4, 2<4>3
RETURN NULL IS NULL, 233 IS NOT NULL, NULL=NULL, NULL<>NULL
RETURN TRUE AND TRUE, TRUE AND FALSE, FALSE OR FALSE, FALSE OR TRUE, NOT FALSE, FALSE XOR FALSE
RETURN CASE WHEN 3>2 THEN '233' ELSE '322' END
```

```bash
RETURN '2'+'33'
RETURN '233'=~'233', '233'=~'23+'
RETURN '233' STARTS WITH '23', '233' ENDS WITH '33', '233' CONTAINS '33'
```

```bash
RETURN [2,23,233], [2,233]+[233], range(0,3)
RETURN size([2,23,233]), size(range(0,3))
WITH [2,23,233] AS x RETURN x, x[0], x[1], x[2], x[3], x[-1], x[0..-1]
UNWIND [2, 23, 233] AS xs WITH xs AS x WHERE x>20 RETURN x
RETURN 233 IN [233], 33 IN [233]
RETURN [x IN range(0,3)], [x IN range(0,3) WHERE x>1 | x+233]
```

```bash
RETURN {k1:-2, k2: '233', k3:[2,3]}
WITH {p: {q:233}} AS x RETURN x.p.q
```

### PATH

```bash
CREATE (x0:Node {p:'0'}), (x1:Node {p:'1'}), (x2:Node {p:'2'})
CREATE (x0)-[:R {p:'01'}]->(x1)-[:R {p:'12'}]->(x2)-[:R {p:'20'}]->(x0)
CREATE (x0)-[:R {p:'02'}]->(x2)-[:R {p:'21'}]->(x1)-[:R {p:'10'}]->(x0)
RETURN x0,x1,x2

MATCH (x0:Node {p:'0'}), (x1:Node {p:'1'})
MATCH p=(x0)-[:R*..5]->(x1)
RETURN nodes(p)
```

```bash
CREATE (xs:Node {p:'s'}), (xe:Node {p:'e'}),
    (x0:Node {p:'0'}),
    (x1:Node {p:'1'}), (x2:Node {p:'2'}),
    (x3:Node {p:'3'}), (x4:Node {p:'4'}), (x5:Node {p:'5'})
CREATE (xs)-[:R {p:'00'}]->(xe),
    (xs)-[:R {p:'10'}]->(x0)-[:R {p:'11'}]->(xe),
    (xs)-[:R {p:'20'}]->(x1)-[:R {p:'21'}]->(x2)-[:R {p:'22'}]->(xe),
    (xs)-[:R {p:'30'}]->(x3)-[:R {p:'31'}]->(x4)-[:R {p:'32'}]->(x5)-[:R {p:'33'}]->(xe)
RETURN xs,xe,x0,x1,x2,x3,x4,x5

MATCH (xs:Node {p:'s'}), (xe:Node {p:'e'})
MATCH (xs)-[r]->(xe), p=(xs)-[*]->(xe)
RETURN r, relationships(p)

MATCH (xs:Node {p:'s'}), (xe:Node {p:'e'})
MATCH p=(xs)-[:R*3..]->(xe)
RETURN p

MATCH (xs:Node {p:'s'}), (xe:Node {p:'e'})
MATCH p=(xs)-[*]->(xe)
RETURN relationships(p)

MATCH (xs:Node {p:'s'}), (xe:Node {p:'e'})
MATCH p=(xs)-[*]->(xe)
RETURN nodes(p)
```

### MAP PROJECTION

```bash
CREATE (n0:Node {name:'n0'}), (n1:Node {name:'n1'})
CREATE (m0:Mode {name:'m0'}), (m1:Mode {name:'m1'}), (m2:Mode {name:'m2'})
CREATE (n0)-[:R]->(m0), (n0)-[:R]->(m1)
CREATE (n1)-[:R]->(m1), (n1)-[:R]->(m2)

MATCH (x:Node {name:'n0'})-->(y:Mode) RETURN x {.name, yname: collect(y {.name})}
MATCH (x:Node)-->(y:Mode) with x, count(y) AS countY RETURN x {.name, countY}
MATCH (x:Node {name:'n0'}) RETURN x {.*, .id}
```

### UNWIND

see [link](https://neo4j.com/docs/developer-manual/current/cypher/clauses/unwind/)

```bash
UNWIND [1,2,3,NULL] AS x RETURN x

UNWIND [2,3,3] AS x RETURN collect(x)
UNWIND [2,3,3] AS x WITH DISTINCT x RETURN collect(x)

UNWIND [2, [2,23], [2,23,233]] AS x
UNWIND x AS y RETURN y

UNWIND [] AS x RETURN x, 'nothing returned'

WITH [] AS x
UNWIND CASE WHEN size(x)=0 THEN [NULL] ELSE x END AS y
RETURN y, 'null returned'

UNWIND NULL AS x RETURN x

UNWIND [{year:2014,id:1}, {year:2014,id:2}] AS event
MERGE (y:Year {year:event.year})
MERGE (y)<-[:IN]-(e:Event {id:event.id})
```

### FOREACH

```bash
FOREACH (x IN ['n0','n1','n2']| CREATE (:Node {name:x}))
```

### MERGE2

```bash
UNWIND [2,3,3] AS x CREATE (y:Node {name:x}) RETURN y
UNWIND [2,3,3] AS x MERGE (y:Node {name:x}) RETURN y
UNWIND [2,3,3] AS x WITH DISTINCT x CREATE (y:Node {name:x}) RETURN y
```

### CALL YIELD

```bash
CALL `db`.`labels`
CALL dbms.procedures() YIELD name, signature WHERE name='dbms.listConfig' RETURN name, signature
```

### SCHEMA - INDEX and CONSTRAINT

```bash
CALL db.indexes
CREATE INDEX ON :Node(name)
DROP INDEX ON :Node(name)
```

```bash
CALL db.constraints
CREATE CONSTRAINT ON (x:Node) ASSERT x.name IS UNIQUE
DROP CONSTRAINT ON (x:Node) ASSERT x.name IS UNIQUE

CREATE CONSTRAINT ON (x:Node) ASSERT exists(x.name)
DROP CONSTRAINT ON (x:Node) ASSERT exists(x.name)

CREATE CONSTRAINT ON ()-[x:R]->() ASSERT exists(x.name)
DROP CONSTRAINT ON ()-[x:R]->() ASSERT exists(x.name)

CREATE CONSTRAINT ON (x:Node) ASSERT (x.name, x.id) IS NODE KEY
DROP CONSTRAINT ON (x:Node) ASSERT (x.name, x.id) IS NODE KEY
```

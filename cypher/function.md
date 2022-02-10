# cypher function

## predicate function

see [link-3.4.1](https://neo4j.com/docs/developer-manual/current/cypher/functions/predicate/)

```bash
WITH [2,3,4] AS x RETURN all(y in x WHERE y>0), all(y in x WHERE y>3)
WITH [2,3,4] AS x RETURN any(y in x WHERE y>4), any(y in x WHERE y>3)
WITH [2,3,4] AS x RETURN none(y in x WHERE y>4), none(y in x WHERE y>3)
WITH [2,3,4] AS x RETURN single(y in x WHERE y>4), single(y in x WHERE y>3)

CREATE (n0:Node {name:'n0'}) -[:R]-> (n1:Node {name:'n1'})
MATCH (x:Node) WHERE exists((x)-->()) RETURN x
MATCH (x:Node) RETURN x, exists((x)-->())
```

## scalar function

see [link-3.4.2](https://neo4j.com/docs/developer-manual/current/cypher/functions/scalar/#functions-size)

```bash
CREATE (:Node {name:'n1'}) -[:R]-> (:Node {name:'n2'}) -[:R]-> (:Node {name:'n3'})
MATCH p=(:Node {name:'n1'}) -[:R*]-> (:Node {name:'n3'})
RETURN size('233'), size([2,3,3]), size(()--()), length(p)
```

```bash
CREATE (:Node {name:'n1'}) -[:R]-> (:Node {name:'n2'})
MATCH ()-[r]->() RETURN startNode(r), endNode(r), type(r)
```

```bash
CREATE (:Node {name:'n1'})
MATCH (x) RETURN properties(x)
RETURN properties({a:-1, b:-2})
```

```bash
RETURN coalesce(NULL), coalesce(NULL,233), coalesce(233,NULL), coalesce(23,3)
UNWIND [2,NULL,3] AS x RETURN coalesce(x,3)
RETURN head([2,3,3]), last([2,3,3])
RETURN randomUUID()

RETURN toBoolean('TRUE'), toBoolean('FALSE'), toBoolean('1'), toBoolean('0'), toBoolean('something')
RETURN toFloat('2.33'), toFloat(233)
RETURN toInteger('233'), toInteger('2.33')
```

## aggregating function

see [link-3.4.3](https://neo4j.com/docs/developer-manual/current/cypher/functions/aggregating/)

TODO

## list function

see [link-3.4.4](https://neo4j.com/docs/developer-manual/current/cypher/functions/list/)

```bash
RETURN range(0,10), range(0,10,3), range(10,0,-3)
RETURN reverse([2,3,3]), head([2,3,3]), tail([2,3,3])

RETURN extract(x in range(0,3)| x+233)
RETURN filter(x in range(0,3) WHERE x>2)
RETURN reduce(x0=1, x IN [2,3,3]| x0*10+x)
```

```bash
CREATE (:Node:Mode {name:'nm0'}) -[:R]-> (:Node {name:'n0'}) -[:Mode]-> (:Mode {name:'m0'})
MATCH p=(x:Node:Mode)-[*]->(y:Mode)
RETURN keys({a:1,b:2}), keys(x), labels(x), labels(y), nodes(p), relationships(p)
```

## mathematical function

see [link-3.4.5](https://neo4j.com/docs/developer-manual/current/cypher/functions/mathematical/numeric/)

```bash
RETURN abs(-2.33), abs(0), abs(2.33), abs(NULL)
RETURN ceil(2), ceil(2.33), ceil(3), ceil(NULL)
RETURN floor(2), floor(2.33), floor(3), floor(NULL)
RETURN rand() AS x, rand() AS y, rand() AS z
RETURN round(2.4), round(2.5), round(2.6), round(3.5), round(NULL)
RETURN sign(-1), sign(0), sign(1), sign(NULL)
```

```bash
RETURN e(), e()^2, exp(1), exp(2)//2.718281828459045
RETURN log(exp(1)), log(exp(2))
RETURN log10(10^2.33)
RETURN sqrt(233^2)
```

```bash
RETURN pi(), degrees(pi()), radians(degrees(pi()))
RETURN cos(pi()), acos(cos(0.233))
RETURN sin(pi()), asin(sin(0.233))
RETURN tan(pi()/2), atan(tan(0.233)), tan(atan2(2,1))
RETURN tan(0.233)*cot(0.233)
```

## string function

see [link-3.4.8](https://neo4j.com/docs/developer-manual/current/cypher/functions/string/)

```bash
RETURN lTrim('\t \n\r233'), rTrim('233\r\n\t ')
RETURN replace('2333','233','23')
RETURN reverse('233')
RETURN left('233',2), right('233',2), substring(' 233 ',1,3)
RETURN split('2 3 3', ' ')
RETURN toUpper('abc'), toLower('ABC')
RETURN toString(233), toString(2.33), toString(False), toString(NULL)
```

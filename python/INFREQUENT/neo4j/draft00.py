from neo4j import GraphDatabase

driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','xxxx'))

# with session.begin_transaction() as tx: pass
def add_friend(tx, name, friend_name):
    cmd = [
        'MERGE (a:Person {name: $name})',
        'MERGE (a) -[:KNOWS]->(b:Person {name:$friend_name})',
    ]
    tx.run('\n'.join(cmd), name=name, friend_name=friend_name)

def print_friends(tx, name):
    cmd = [
        'MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name=$name',
        'RETURN friend.name ORDER BY friend.name',
    ]
    return list(tx.run('\n'.join(cmd), name=name))

def rm_rf(tx):
    if input('type Y to confirm REMOVE ALL: Y/[N]')=='Y':
        tx.run('MATCH (n) DETACH DELETE (n)')

with driver.session() as sess:
    sess.write_transaction(rm_rf)
    sess.write_transaction(add_friend, 'Arthur', 'Guinevere')
    sess.write_transaction(add_friend, 'Arthur', 'Lancelot')
    sess.write_transaction(add_friend, 'Arthur', 'Merlin')
    z1 = sess.read_transaction(print_friends, 'Arthur')
    for x in z1:
        print(x['friend.name'])

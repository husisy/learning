import os
import rdflib
from rdflib.namespace import DC, FOAF
from rdflib import Graph, Literal, BNode, Namespace, RDF, URIRef

z1 = Graph()
z1.load('http://dbpedia.org/resource/Semantic_Web')
for s,p,o in z1:
    print(s,p,o)

semweb = rdflib.URIRef('http://dbpedia.org/resource/Semantic_Web')
x1 = z1.value(semweb, rdflib.RDFS.label)

dbpedia = rdflib.Namespace('http://dbpedia.org/ontology/')
x1 = [x for x in z1.objects(semweb, dbpedia['abstract']) if x.language=='en']

z1 = Graph()
x1 = z1.parse('http://www.w3.org/People/Berners-Lee/card')
print('graph has {} statements'.format(len(z1)))
for s,p,o in z1:
    assert (s,p,o) in z1
print(z1.serialize(format='n3').decode('utf-8'))

z1 = Graph()
x1 = BNode()
z1.add((x1, RDF.type, FOAF.Person))
z1.add((x1, FOAF.nick, Literal('donna', lang='foo')))
z1.add((x1, FOAF.name, Literal('Donna Fales')))
z1.add((x1, FOAF.mbox, URIRef('mailto:donna@example.ort')))
for s,p,o in z1:
    print(s,p,o)
for person in z1.subjects(RDF.type, FOAF.Person):
    for mbox in z1.objects(person, FOAF.mbox):
        print(mbox)
z1.bind('dc', DC)
z1.bind('foaf', FOAF)
print(z1.serialize(format='n3').decode('utf-8'))

z1 = Graph()
# z1.parse('finance-data.nt', format='nt')
z1.parse('http://bigasterisk.com/foaf.rdf')
for s,p,o in z1:
    print(s,p,o)


bob = URIRef('http://2333.org/people/Bob')
linda = BNode()
age = Literal(24)
height = Literal(76.5)
ns1 = Namespace('http://2333.org/people/')
print(ns1.bob, ns1.linda, ns1.eve)
print(RDF.type, FOAF.knows)
z1 = Graph()
z1.add((bob, RDF.type, FOAF.Person))
z1.add((bob, FOAF.name, Literal('Bob')))
z1.add((bob, FOAF.knows, linda))
z1.add((linda, RDF.type, FOAF.Person))
z1.add((linda, FOAF.name, Literal('Linda')))
print(z1.serialize(format='turtle').decode('utf-8'))

z1.add((bob, FOAF.age, Literal(42)))
print("Bob's age: {}".format(z1.value(bob, FOAF.age)))
z1.set((bob, FOAF.age, Literal(43)))
print("Bob's age: {}".format(z1.value(bob, FOAF.age)))

# fail
# z1.parse('http://danbri.livejournal.com/data/foaf')
# for s,_,n in z1.triples((None,FOAF['member_name'],None)):
#     z1.add((s,FOAF['member_name'],n))

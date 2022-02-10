import dns.resolver
import dns.zone

z0 = dns.resolver.resolve('dnspython.org', 'MX')
for x in z0:
    print(f'Host: {x.exchange}; preference: {x.preference}')


z = dns.zone.from_xfr(dns.query.xfr('10.0.0.1', 'dnspython.org'))
names = z.nodes.keys()
names.sort()
for n in names:
    print(z[n].to_text(n))

# shenmegui....

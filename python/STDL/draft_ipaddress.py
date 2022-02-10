import ipaddress
import socket
import random


hf_ipv6_expand = lambda x: ipaddress.IPv6Address(x).exploded
hf_ipv6_reduce = lambda x: ipaddress.IPv6Address(x).compressed


def hf_one_int_to_ip_bin(x, family='ipv4'):
    assert family in {'ipv4','ipv6'}
    ret = []
    for _ in range(4 if family=='ipv4' else 16):
        if x>0:
            ret.append(x%256)
            x = x // 256
        else:
            ret.append(0)
    assert x==0
    ret = bytes(ret[::-1])
    return ret


def ip_str_to(ip_str, dtype='int'):
    assert dtype in {'int', 'bin'}
    assert ('.' in ip_str) or (':' in ip_str)
    if ('.' in ip_str) and (dtype=='int'):
        ret = [int(x) for x in ip_str.split('.')]
    elif ('.' in ip_str) and (dtype=='bin'):
        ret = bytes([int(x) for x in ip_str.split('.')])
    elif (':' in ip_str) and (dtype=='int'):
        ret = [int(x,16) for x in hf_ipv6_expand(ip_str).split(':')]
    else:
        tmp0 = [int(x,16) for x in hf_ipv6_expand(ip_str).split(':')]
        ret = bytes([y for x in tmp0 for y in [x//256,x%256]])
    return ret


def ip_int_to(ip_int, dtype='str'):
    assert dtype in {'str', 'bin'}
    len_ip_int = len(ip_int)
    assert len_ip_int in {4,8}
    if (len_ip_int==4) and dtype=='str':
        ret = '.'.join(str(x) for x in ip_int)
    elif (len_ip_int==4) and dtype=='bin':
        ret = bytes(ip_int)
    elif (len_ip_int==8) and dtype=='str':
        ret = hf_ipv6_reduce(':'.join(hex(x)[2:] for x in ip_int))
    else:
        ret = bytes([y for x in ip_int for y in [x//256,x%256]])
    return ret


def ip_bin_to(ip_bin, dtype='str'):
    assert dtype in {'str','int'}
    len_ip_bin = len(ip_bin)
    assert len_ip_bin in {4,16}
    if (len_ip_bin==4) and dtype=='str':
        ret = '.'.join(str(x) for x in ip_bin)
    elif (len_ip_bin==4) and dtype=='int':
        ret = list(ip_bin)
    elif (len_ip_bin==16) and dtype=='str':
        tmp0 = [x*256+y for x,y in zip(ip_bin[::2],ip_bin[1::2])]
        ret = hf_ipv6_reduce(':'.join(hex(x)[2:] for x in tmp0))
    else:
        ret = [x*256+y for x,y in zip(ip_bin[::2],ip_bin[1::2])]
    return ret


def random_ip(family='ipv4', dtype='str'):
    assert family in {'ipv4', 'ipv6'}
    assert dtype in {'int', 'str', 'bin'}
    if family=='ipv4':
        ip_int = [random.randint(0,255) for _ in range(4)]
        if dtype=='int':
            ret = ip_int
        elif dtype=='str':
            ret = '.'.join(str(x) for x in ip_int)
        else:
            ret = bytes(ip_int)
    else:
        ip_int = [random.randint(0,2**16-1) for _ in range(8)]
        if dtype=='int':
            ret = ip_int
        elif dtype=='str':
            ret = ':'.join(hex(x)[2:] for x in ip_int)
        else:
            ret = bytes([y for x in ip_int for y in [x//256,x%256]])
    return ret


def test_one_int_to_ip():
    x0 = 42540766411282592856903984951653826561
    x1 = ip_bin_to(hf_one_int_to_ip_bin(x0, 'ipv6'), 'str')
    x2 = str(ipaddress.IPv6Address(x0))
    assert x1==x2


#TODO https://docs.python.org/3/howto/ipaddress.html#ipaddress-howto

x0 = ipaddress.ip_address('192.0.2.1')
str(x0)
int(x0)
x0.version #4
x0.packed
x1 = ipaddress.ip_address('2001:db8::1')
x1.version #6
x1.exploded #str('2001:0db8:0000:0000:0000:0000:0000:0001')
x1.compressed #str('2001:db8::1')
# ipaddress.IPv4Address
# ipaddress.IPv6Address


x0 = ipaddress.ip_network('192.0.2.0/24')
x0.num_addresses #256
x0.netmask #IPv4Address
x0.hostmask
list(x0.hosts()) #len()=254, from 192.0.2.1 to 192.0.2.254
x0[0]
# ipaddress.ip_network('192.0.2.1/24') #ValueError
ipaddress.ip_network('192.0.2.1/24', strict=False)
x1 = ipaddress.ip_network('2001:db8::0/96')
x1.exploded #str('2001:0db8:0000:0000:0000:0000:0000:0000/96')
x1.compressed #str('2001:db8::/96')


ipaddress.ip_address('192.0.2.1') in ipaddress.ip_network('192.0.2.0/24')
ipaddress.ip_address('192.0.2.1') < ipaddress.ip_address('192.0.2.2')


x0 = ipaddress.ip_interface('192.0.2.1/24')
x0.network #IPv4Network
ipaddress.ip_interface('2001:db8::1/96')

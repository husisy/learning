import time
import socket
import random
import ipaddress
import multiprocessing

hf_ipv6_expand = lambda x: ipaddress.IPv6Address(x).exploded
hf_ipv6_reduce = lambda x: ipaddress.IPv6Address(x).compressed


def demo_request_html():
    print('# demo_request_html')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('www.sina.com.cn', 80))
    sock.send(b'GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')
    data = []
    while True:
        tmp0 = sock.recv(1024)
        if tmp0:
            data.append(tmp0)
        else:
            break
    sock.close()
    data = (b'').join(data)
    header,body = data.split(b'\r\n\r\n', 1)
    print('[demo_request_html].header:\n', header.decode('utf-8'))
    print('[demo_request_html].body:\n:', body.decode('utf-8'))


def _tcp_server_kernel(sock, addr):
    print('[tcp-server], connect to {}'.format(addr))
    sock.send(b'welcome')
    while True:
        data = sock.recv(1024).decode('utf-8') #block until CLIENT.send()
        print('[tcp-server] receive "{}"'.format(data))
        if not data or data=='exit':
            break
        sock.send('hi {}'.format(data).encode('utf-8'))
        # time.sleep(0.1) #unnecessary
    sock.close()

def _tcp_server(port):
    ip = '127.0.0.1'
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((ip, port))
    sock.listen(5) #max connection
    sock_i, addr = sock.accept() #block until CLIENT.connect()
    # addr is a tuple of length 2, (str,int), ('127.0.0.1', 45956)
    _tcp_server_kernel(sock_i, addr)
    sock.close()

def _tcp_client(port):
    ip = '127.0.0.1'
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((ip, port)) #error if no SERVER.listen
    data = sock.recv(1024).decode('utf-8')
    print('[tcp-client] receive:', data)
    for x in [b'Alice', b'Bob']:
        _ = sock.send(x) #not block, return len()
        data = sock.recv(1024).decode('utf-8') #block until SERVER.send()
        print('[tcp-client] receive "{}"'.format(data))
    sock.send(b'exit')
    sock.close()

def demo_tcp_server_client():
    '''
    clent --[connect]-->  server
    clent <--welcome----  server
    clent --Alice------>  server
    clent <--hi Alice---  server
    clent --Bob-------->  server
    clent <--hi Bob-----  server
    clent --exit-------->  server
    '''
    port = 23333
    process_list = [
        multiprocessing.Process(target=_tcp_server, args=(port,)),
        multiprocessing.Process(target=_tcp_client, args=(port,))
    ]
    process_list[0].start()
    time.sleep(0.1) #wait 0.1 second for server to set up
    process_list[1].start()
    for x in process_list:
        x.join()


def _udp_server(port):
    ip = '127.0.0.1'
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))
    while True: #maybe loop forever, since 'exit' is not guarentee to be received
        data,addr = sock.recvfrom(1024)
        data = data.decode('utf-8')
        print('[udp-server] receive "{}"'.format(data))
        if data=='exit':
            break
        sock.sendto('hi {}'.format(data).encode('utf-8'), addr)
    sock.close()

def _udp_client(port):
    ip = '127.0.0.1'
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for ind0 in range(3):
        sock.sendto('Alice{}'.format(ind0).encode('utf-8'), (ip,port))
        data = sock.recv(1024).decode('utf-8')
        print('[udp-client] receive "{}"'.format(data))
    for _ in range(100): #try to close server
        sock.sendto(b'exit', (ip,port))
    sock.close()

def demo_udp_server_client():
    '''
    clent --Alice0----->  server
    clent <--hi Alice0--  server
    clent --Alice1----->  server
    clent <--hi Alice1--  server
    clent --Alice2----->  server
    clent <--hi Alice2--  server
    clent --exit-------->  server
    clent --exit-------->  server
    '''
    port = 23333
    process_list = [
        multiprocessing.Process(target=_udp_server, args=(port,)),
        multiprocessing.Process(target=_udp_client, args=(port,))
    ]
    process_list[0].start()
    time.sleep(0.1) #wait 0.1 second for server to set up
    process_list[1].start()
    for x in process_list:
        x.join()


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
            ret = hf_ipv6_reduce(':'.join(hex(x)[2:] for x in ip_int))
        else:
            ret = bytes([y for x in ip_int for y in [x//256,x%256]])
    return ret


def demo_ip_int_str_bin():
    x0 = random_ip(family='ipv4', dtype='str')
    x1 = ip_bin_to(ip_int_to(ip_str_to(x0, 'int'), 'bin'), 'str')
    x2 = ip_int_to(ip_bin_to(ip_str_to(x0, 'bin'), 'int'), 'str')
    assert (x0==x1) and (x0==x2)

    x0 = random_ip(family='ipv6', dtype='str')
    x1 = ip_bin_to(ip_int_to(ip_str_to(x0, 'int'), 'bin'), 'str')
    x2 = ip_int_to(ip_bin_to(ip_str_to(x0, 'bin'), 'int'), 'str')
    assert (x0==x1) and (x0==x2)

    x0 = '2001:db8::1'
    x1 = ip_bin_to(ip_int_to(ip_str_to(x0, 'int'), 'bin'), 'str')
    x2 = ip_int_to(ip_bin_to(ip_str_to(x0, 'bin'), 'int'), 'str')
    assert (x0==x1) and (x0==x2)


def demo_socket_pton_ntop():
    ipv4_str = random_ip(family='ipv4', dtype='str')
    x0 = ip_str_to(ipv4_str, 'bin')
    x1 = socket.inet_pton(socket.AF_INET, ipv4_str)
    x2 = socket.inet_aton(ipv4_str)
    assert (x0==x1) and (x0==x2)

    ipv4_bin = random_ip(family='ipv4', dtype='bin')
    x0 = ip_bin_to(ipv4_bin, 'str')
    x1 = socket.inet_ntop(socket.AF_INET, ipv4_bin)
    x2 = socket.inet_ntoa(ipv4_bin)
    assert (x0==x1) and (x0==x2)

    ipv6_str = random_ip(family='ipv6', dtype='str')
    x0 = ip_str_to(ipv6_str, 'bin')
    x1 = socket.inet_pton(socket.AF_INET6, ipv6_str)
    assert x0==x1

    ipv6_bin = random_ip(family='ipv6', dtype='bin')
    x0 = ip_bin_to(ipv6_bin, 'str')
    x1 = socket.inet_ntop(socket.AF_INET6, ipv6_bin)
    assert x0==x1


def demo_free_port():
    sock = socket.socket()
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    print('free port:', port)


if __name__=='__main__':
    demo_request_html()
    demo_tcp_server_client()
    demo_udp_server_client()
    demo_ip_int_str_bin()
    demo_socket_pton_ntop()
    demo_free_port()

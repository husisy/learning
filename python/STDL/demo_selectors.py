import time
import random
import string
import socket
import selectors
import types
import multiprocessing



def _basic_selectors_server_socket_accept(key, mask, sel):
    sock = key.fileobj
    sock_i,addr = sock.accept()
    print(f'[server] connecting {addr}')
    sock_i.setblocking(False)
    data = types.SimpleNamespace(addr=addr, recvbuffer=b'', sendbuffer=b'',
            callback=_basic_selectors_server_socket_i_accept)
    sel.register(sock_i, selectors.EVENT_READ|selectors.EVENT_WRITE, data=data)

def _basic_selectors_server_socket_i_accept(key, mask, sel):
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)
        if recv_data:
            print(f'[server] receiving {recv_data} from {data.addr[1]}')
            data.recvbuffer += recv_data
        else:
            print(f'[server] closing connection to {data.addr[1]}')
            sel.unregister(sock)
            sock.close()
            return 'tag_close'
    if mask & selectors.EVENT_WRITE:
        if not data.sendbuffer and data.recvbuffer:
            data.sendbuffer = data.recvbuffer.decode('utf-8')[::-1].encode('utf-8')
            data.recvbuffer = b''
        if data.sendbuffer:
            print(f'[server] sending {data.sendbuffer} to {data.addr[1]}')
            num_sent = sock.send(data.sendbuffer)
            data.sendbuffer = data.sendbuffer[num_sent:]

def _basic_selectors_server(port, num_connection):
    ip = '127.0.0.1'
    sel = selectors.DefaultSelector()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((ip, port))
    sock.listen()
    sock.setblocking(False)
    data = types.SimpleNamespace(callback=_basic_selectors_server_socket_accept)
    sel.register(sock, selectors.EVENT_READ, data=data)
    num_close = 0
    while num_close < num_connection:
        events = sel.select(timeout=None)
        for key, mask in events:
            ret = key.data.callback(key, mask, sel)
            if isinstance(ret,str) and ret=='tag_close':
                num_close += 1
    sel.close()

def _basic_selectors_client_connection(key, mask, sel):
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)
        if recv_data:
            print(f'[connid-{data.connid}] receiving {recv_data}')
            data.recv_total += len(recv_data)
        if not recv_data or data.recv_total == data.msg_total:
            print(f'[connid-{data.connid}] closing')
            sel.unregister(sock)
            sock.close()
    if mask & selectors.EVENT_WRITE:
        if not data.sendbuffer and data.message:
            data.sendbuffer = data.message.pop(0)
        if data.sendbuffer:
            print(f'[connid-{data.connid}] sending {data.sendbuffer}')
            num_sent = sock.send(data.sendbuffer)
            data.sendbuffer = data.sendbuffer[num_sent:]

def _basic_selectors_client_generate_string(N0):
    tmp0 = [random.randint(1,3) for _ in range(N0)]
    ret = [[''.join(random.choices(string.ascii_letters, k=5)) for _ in range(x)] for x in tmp0]
    ret = [[y.encode('utf-8') for y in x] for x in ret]
    return ret

def _basic_selectors_client(port, num_connection):
    ip = '127.0.0.1'
    sel = selectors.DefaultSelector()
    message_list = _basic_selectors_client_generate_string(num_connection)
    for connid,message_i in enumerate(message_list):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
        sock.setblocking(False)
        tmp0 = sum(len(m) for m in message_i)
        # msg_total should not specified in this way
        data = types.SimpleNamespace(connid=connid, message=message_i, msg_total=tmp0,
                    recv_total=0, sendbuffer=b'', callback=_basic_selectors_client_connection)
        sel.register(sock, selectors.EVENT_READ|selectors.EVENT_WRITE, data=data)
    while len(sel.get_map()):
        events = sel.select(timeout=0.1)
        for key, mask in events:
            key.data.callback(key, mask, sel)
    sel.close()

def demo_basic_selectors():
    port = 23333
    num_connection = 2
    task_server = multiprocessing.Process(target=_basic_selectors_server, args=(port,num_connection))
    task_server.start()
    time.sleep(0.5)
    task_client = multiprocessing.Process(target=_basic_selectors_client, args=(port,num_connection))
    task_client.start()
    task_server.join()
    task_client.join()

if __name__=='__main__':
    demo_basic_selectors()

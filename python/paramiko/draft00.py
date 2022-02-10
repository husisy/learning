import os
import json
import socket
import base64
import paramiko

# private_info.json is not git-synced, private_info_fake.json is served as a sample
with open('private_info.json') as fid:
    _PRIVATE_INFO = json.load(fid)
with open('/etc/ssh/ssh_host_rsa_key.pub') as fid:
    tmp0 = fid.read().strip().split()[1]
SSH_INFO = {
    'hostname': '127.0.0.1',
    'port': 22,
    'username': _PRIVATE_INFO['localhost-user'],
    'password': _PRIVATE_INFO['localhost-password'],
    'rsa_public_key': tmp0,
    'known_hosts_file': os.path.expanduser(os.path.join('~', '.ssh', 'known_hosts')),
    'private_key_file': os.path.expanduser(os.path.join('~', '.ssh', 'id_rsa')),
}



def detect_server_host_key(hostname, port):
    # see https://github.com/paramiko/paramiko/blob/master/demos/demo.py
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((hostname, port))
    t = paramiko.Transport(sock)
    t.start_client()
    ret = t.get_remote_server_key()
    t.close()
    sock.close()
    return ret


def demo_detect_server_host_key():
    z0 = detect_server_host_key('127.0.0.1', 22)
    tmp0 = z0.get_name().split('-',1)[1] #ed25519
    with open(f'/etc/ssh/ssh_host_{tmp0}_key.pub') as fid:
        z1 = fid.read().strip().split()[1]
    assert z0.get_base64() == z1


def demo_add_host_keys_mannually():
    hostname = SSH_INFO['hostname']
    port = SSH_INFO['port']
    username = SSH_INFO['username']
    password = SSH_INFO['password']
    rsa_public_key = paramiko.RSAKey(data=base64.b64decode(SSH_INFO['rsa_public_key']))

    client = paramiko.SSHClient()
    client.get_host_keys().add(hostname, 'ssh-rsa', rsa_public_key)
    client.connect(hostname, port, username, password)
    stdin, stdout, stderr = client.exec_command('ls')
    for line in stdout:
        print('... ' + line.strip())
    client.close()


def demo_connect_with_password():
    hostname = SSH_INFO['hostname']
    port = SSH_INFO['port']
    username = SSH_INFO['username']
    password = SSH_INFO['password']
    client = paramiko.SSHClient()
    client.get_host_keys().load(SSH_INFO['known_hosts_file'])
    client.connect(hostname, port, username, password)
    stdin, stdout, stderr = client.exec_command('ls')
    for line in stdout:
        print('... ' + line.strip())
    client.close()


def demo_connect_with_private_key():
    hostname = SSH_INFO['hostname']
    port = SSH_INFO['port']
    username = SSH_INFO['username']
    client = paramiko.SSHClient()
    client.get_host_keys().load(SSH_INFO['known_hosts_file'])
    client.connect(hostname, port, username, key_filename=SSH_INFO['private_key_file'])
    stdin, stdout, stderr = client.exec_command('ls')
    for line in stdout:
        print('... ' + line.strip())
    client.close()

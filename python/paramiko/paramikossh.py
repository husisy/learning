import os
import sys
import base64
import getpass
import socket
import argparse
import traceback

import paramiko

# not support windows
import select
import termios
import tty

hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

def posix_shell(chan):
    oldtty = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        chan.settimeout(0.0)

        print('[zc-info] chan', chan)
        print('[zc-info] sys.stdin', sys.stdin)
        while True:
            r, _, _ = select.select([chan, sys.stdin], [], [])
            assert len(r)==1 and (r[0]==chan) or (r[0]==sys.stdin)
            print('[zc-info] r', r)
            if chan in r:
                try:
                    x = paramiko.py3compat.u(chan.recv(1024))
                    if len(x) == 0:
                        sys.stdout.write("\r\n")
                        break
                    sys.stdout.write(x)
                    sys.stdout.flush()
                except socket.timeout:
                    pass
            if sys.stdin in r:
                x = sys.stdin.read(1)
                if len(x) == 0:
                    break
                chan.send(x)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, oldtty)


parser = argparse.ArgumentParser(description='obs command line utility')
parser.add_argument('host', type=str)
parser.add_argument('-p', '--port', default=22, type=int)
parser.add_argument('-i', '--identity-file', default=None, help='identity file')
parser.add_argument('--password', default=None)
args = parser.parse_args()
assert '@' in args.host
args.username, args.hostname = args.host.split('@')
if ':' in args.hostname:
    args.hostname, args.port = args.hostname.split(':')
    args.port = int(args.port)
if args.identity_file is None:
    tmp0 = os.path.expanduser(os.path.join('~', '.ssh', 'id_rsa'))
    if os.path.exists(tmp0):
        args.identity_file = tmp0

paramiko.util.log_to_file(hf_file('paramikossh.log'))
if (args.identity_file is None) and (args.password is None):
    args.password = getpass.getpass(f"Password for {args.username}@{args.hostname}: ")

try:
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())
    client.connect(args.hostname, args.port, args.username, password=args.password, key_filename=args.identity_file)

    chan = client.invoke_shell()
    posix_shell(chan)
    chan.close()
    client.close()

except Exception as e:
    print("*** Caught exception: %s: %s" % (e.__class__, e))
    traceback.print_exc()
    try:
        client.close()
    except:
        pass
    sys.exit(1)

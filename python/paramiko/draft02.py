# implement bash-python middleware
import os
import sys
import select
import subprocess

import termios
import tty

# TODO https://pydoc.net/paraproxy/1.2/paraproxy/ paramiko paraproxy
# TODO https://stackoverflow.com/q/10488832

def posix_shell(chan):
    oldtty = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        chan.settimeout(0.0)

        while True:
            r, _, _ = select.select([chan, sys.stdin], [], [])
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


# maybe bufsize = 0?
# https://stackoverflow.com/a/20458111
def bash():
    stdin_pipe_r, stdin_pipe_w = os.pipe()
    stdout_pipe_r, stdout_pipe_w = os.pipe()
    proc = subprocess.Popen(['bash'], stdin=stdin_pipe_r, stdout=stdout_pipe_w, stderr=stdout_pipe_w)

    oldtty = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())

        while True:
            r, _, _ = select.select([stdout_pipe_r, sys.stdin], [], [])
            print('[zc-debug/line24]', r)
            if r[0]==stdout_pipe_r:
                # strange, nev
                x = os.read(stdout_pipe_r, 1024)
                print('[zc-debug/line26]', x)
                if len(x)==0:
                    # TODO close
                    break
                sys.stdout.write(x.decode('utf-8'))
                sys.stdout.flush()
            else:
                x = sys.stdin.read(1)
                if len(x) == 0:
                    # TODO close
                    break
                print('[zc-debug/line37]', x)
                os.write(stdin_pipe_w, x.encode('utf-8')) #TODO maybe return value not correct?
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, oldtty)
        for x in [stdin_pipe_r,stdin_pipe_w,stdout_pipe_r,stdout_pipe_w]:
            os.close(x)
    proc.communicate()

# https://docs.python.org/3/library/select.html
# https://docs.python.org/3/library/os.html#os.pipe
# https://stackoverflow.com/a/6050722


import os
import time
import shlex
import tempfile
import subprocess


hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file()):
    os.makedirs(hf_file())

with open(hf_file('tbd00.sh'), 'w') as fid:
    fid.write('sleep 1s\n echo "hello world"\n')

stdout_pipe_r, stdout_pipe_w = os.pipe()
proc = subprocess.Popen(['bash', hf_file('tbd00.sh')], stdout=stdout_pipe_w)
z0 = os.read(stdout_pipe_r, 1024).decode()
os.close(stdout_pipe_r)
os.close(stdout_pipe_w)


PASSPHRASE='...'

in_fd,out_fd=os.pipe()
os.write(out_fd,PASSPHRASE)
os.close(out_fd)
cmd='gpg --passphrase-fd {fd} -c'.format(fd=in_fd)
with open('filename.txt','r') as stdin_fh:
    with open('filename.gpg','w') as stdout_fh:
        proc=subprocess.Popen(shlex.split(cmd),
                              stdin=stdin_fh,
                              stdout=stdout_fh)
        proc.communicate()
os.close(in_fd)

fid_out = tempfile.TemporaryFile()
proc = subprocess.Popen(['ls'], stdout=fid_out)

in_fd,out_fd = os.pipe()

fid_in = tempfile.TemporaryFile()
fid_out = tempfile.TemporaryFile()



proc = subprocess.Popen('md5sum', stdout=subprocess.PIPE, stdin=subprocess.PIPE)

#!/usr/bin/env python
text = 'hello'
proc = subprocess.Popen('md5sum', stdout=subprocess.PIPE, stdin=subprocess.PIPE)
proc.stdin.write(text.encode()) #byte
proc.stdin.close()
result = proc.stdout.read() #byte
print(result)

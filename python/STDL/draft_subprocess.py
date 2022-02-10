import os
import sys
import time
import shlex
import random
import signal
import subprocess


def demo_subprocess_run():
    tmp0 = round(random.uniform(0.233, 0.466), 3)
    print(f'sleep for {tmp0} seconds')
    time0 = time.time()
    subprocess.run(['sleep', f'{tmp0}s']) #block until subprocess finish
    print(f'actually pass {time.time() - time0} seconds')


def demo_split_shell():
    tmp0 = '''
    /bin/vikings -input eggs.txt -output "spam spam.txt" -cmd "echo '$MONEY'"
    '''
    shlex.split(tmp0.strip())


def demo_start_process_non_blocking():
    # https://stackoverflow.com/a/16071877
    time0 = time.time()
    print('start process "sleep 0.5" without blocking')
    proc = subprocess.Popen(['sleep', '0.5'])
    # while proc.poll() is None:
    while True:
        poll_ret = proc.poll() #if not call proc.poll(), proc.returncode will always be "None"
        print(f'poll_ret is "{poll_ret}", proc.returncode is "{proc.returncode}"')
        if poll_ret is not None:
            break
        tmp0 = round(time.time() - time0, 2)
        print(f'process still running, {tmp0} seconds')
        time.sleep(0.1)
    tmp0 = round(time.time() - time0, 2)
    print(f'process exited with returncode "{proc.returncode}", {tmp0} seconds')


def demo_kill_nonblocking_process():
    # https://stackoverflow.com/a/4791612
    print('start process "sleep 0.5" without blocking')
    time0 = time.time()
    proc = subprocess.Popen(['sleep', '0.5'], preexec_fn=os.setsid)

    tmp0 = round(time.time()-time0, 2)
    print(f'main process sleep 0.2 seconds, {tmp0} seconds')
    time.sleep(0.2)

    tmp0 = round(time.time()-time0, 2)
    print(f'main process kill subprocess, {tmp0} seconds')
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)  # Send the signal to all the process groups

    tmp0 = round(time.time()-time0, 2)
    print(f'subprocess killed with returncode "{proc.returncode}", {tmp0} seconds')

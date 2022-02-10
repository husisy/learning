import os
import time
import signal
import multiprocessing
import functools


def hf_print_signal(signum, stack, pid=None):
    print(f'[pid={pid}] receive: {signum}')


def demo_misc00():
    tmp0 = (x for x in dir(signal) if x.startswith('SIG') and '_' not in x)
    for x in tmp0:
        print(x, ':', getattr(signal, x))
    # signal.SIG_DFL: default 0
    # signal.SIG_IGN: ignore 1
    # signal.SIGINT: interruption 2
    # signal.SIGUSR1 10
    # signal.SIGUSR2 12

    # signal.getsignal()


# TODO get the regiested function handle
# TODO restore the original function handle
def demo_basic():
    pid = os.getpid()
    print('you can try "kill -USR1 {}"'.format(pid))
    print('you can try "kill -USR2 {}"'.format(pid))
    print('you can try "kill -INT {}"'.format(pid)) #same as "kill <$pid>"

    time_wait = 30
    hf0 = functools.partial(hf_print_signal, pid=pid)
    signal.signal(signal.SIGUSR1, hf0)
    signal.signal(signal.SIGUSR2, hf0)
    print(f'waiting {time_wait} seconds', end='', flush=True)
    for _ in range(time_wait):
        time.sleep(1)
        print('.', end='', flush=True)
    print()


def _communication_receive_signal(queue, time_wait=5):
    pid = os.getpid()
    print(f'[receiver] pid={pid}')
    queue.put(pid)

    hf0 = functools.partial(hf_print_signal, pid=pid)
    signal.signal(signal.SIGUSR1, hf0)
    signal.signal(signal.SIGUSR2, hf0)

    try:
        for _ in range(time_wait):
            time.sleep(1)
    except KeyboardInterrupt:
        print(f'[pid={pid}] receive KeyboardInterrupt signal')

def _communication_send_signal(pid_receiver):
    time.sleep(1)
    os.kill(pid_receiver, signal.SIGUSR1)
    time.sleep(1)
    os.kill(pid_receiver, signal.SIGUSR2)
    time.sleep(1)
    os.kill(pid_receiver, signal.SIGINT) #will cause receiver KeyboardInterrupt

def demo_communication():
    queue = multiprocessing.Queue()
    p_receive = multiprocessing.Process(target=_communication_receive_signal, args=(queue,))
    p_receive.start()
    pid_receiver = queue.get()
    p_send = multiprocessing.Process(target=_communication_send_signal, args=(pid_receiver,))
    p_send.start()
    p_receive.join()
    p_send.join()


if __name__=='__main__':
    demo_misc00()
    # demo_basic()
    # demo_communication()

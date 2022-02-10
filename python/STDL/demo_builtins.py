import builtins
import concurrent.futures


def suppress_print(is_suppress=True):
    """Suppress printing on the current device. Force printing with `force=True`."""
    builtins_print = builtins.print
    builtins.print('', end='', flush=True) #flush first
    def print(*args, **kwargs):
        force = False
        if 'force' in kwargs:
            force = kwargs.pop('force')
        if (not is_suppress) or force:
            builtins_print(*args, **kwargs)
    builtins.print = print
    builtins.print.builtins_print = builtins_print #for restore usage

def restore_print():
    if hasattr(builtins.print, 'builtins_print'):
        builtins.print = builtins.print.builtins_print

def _suppress_print_proc_i(rank):
    print(f'[rank={rank}] before suppress_print()')
    suppress_print(rank!=0)
    print(f'[rank={rank}] after suppress_print()')
    print(f'[rank={rank}][force=True] after suppress_print()', force=True)
    restore_print()
    print(f'[rank={rank}] after restore_print()')

def demo_suppress_print():
    world_size = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=world_size) as executor:
        job_list = [executor.submit(_suppress_print_proc_i, x) for x in range(world_size)]
        for x in job_list:
            x.result()

if __name__=='__main__':
    demo_suppress_print()

# see https://pytorch.org/docs/stable/notes/cuda.html
import os
import time
import numpy as np
import torch
import concurrent.futures

if os.environ.get('CUDA_VISIBLE_DEVICES') is not None:
    del os.environ['CUDA_VISIBLE_DEVICES']


def demo_cuda_device():
    cuda = torch.device('cuda') #default cuda device, depend on context manager
    cuda0 = torch.device('cuda:0') #the first CUDA_VISIBLE_DEVICES if set
    cuda2 = torch.device('cuda:2') #fail when len(CUDA_VISIBLE_DIVECES)<2

    x = torch.tensor([2,23,233]).cuda() #cuda:0
    x = torch.tensor([2,23,233], device=cuda0) #cuda:0

    with torch.cuda.device(1):
        z0 = torch.tensor([2,23,233]).cuda() #cuda:1
        z1 = torch.tensor([2,23,233], device=cuda) #cuda:1
        z2 = torch.tensor([1., 2.]).to(device=cuda) #cuda:1
        z3 = z0 + z1 #cuda:1, could be out of context manager

        z4 = torch.tensor([2,23,233], device=cuda2) #cuda:2
        z5 = torch.tensor([2,23,233]).to(cuda2) #cuda:2
        z6 = torch.randn(2).cuda(cuda2) #cuda:2
        z7 = z4 + z5 #cuda:2, could be out of context manager


def demo_cuda_event_timing(N0=1024, N1=32768, N2=2048, num_repeat=10):
    cuda = torch.device('cuda')
    torch0 = torch.randn(N0, N1, dtype=torch.float32, device=cuda)
    torch1 = torch.randn(N1, N2, dtype=torch.float32, device=cuda)
    time_list = []
    for _ in range(num_repeat+1):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        t0 = time.time()
        torch2 = torch.matmul(torch0, torch1) + 233
        end_event.record()
        torch.cuda.synchronize()
        time_list.append(start_event.elapsed_time(end_event))
    time_list = np.array(time_list[1:]) / 1000 #convert milisecond to second
    tmp0 = time_list.std() / np.sqrt(time_list.size-1)
    print('event-timing: {} ± {}'.format(time_list.mean(), tmp0))


def hf0_demo_CUDA_LAUNCH_BLOCKING(N0=1024, N1=32768, N2=2048, num_repeat=10, tag_add_environ=True):
    if tag_add_environ:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    cuda = torch.device('cuda')
    torch0 = torch.randn(N0, N1, dtype=torch.float32, device=cuda)
    torch1 = torch.randn(N1, N2, dtype=torch.float32, device=cuda)
    time_list = []
    for _ in range(num_repeat+1):
        t0 = time.time()
        torch2 = torch.matmul(torch0, torch1) + 233
        time_list.append(time.time() - t0) #WARNING, this is NOT correct timing, just to demo CUDA_LAUNCH_BLOCKING
    time_list = np.array(time_list[1:]) #drop first
    return time_list

def demo_CUDA_LAUNCH_BLOCKING():
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        time_non_block = executor.submit(hf0_demo_CUDA_LAUNCH_BLOCKING, tag_add_environ=False).result()
        time_block = executor.submit(hf0_demo_CUDA_LAUNCH_BLOCKING, tag_add_environ=True).result()

    tmp0 = time_non_block.std() / np.sqrt(time_non_block.size-1)
    print('time-info(non-block): {:.4} ± {:.4}'.format(time_non_block.mean(), tmp0))

    tmp0 = time_block.std() / np.sqrt(time_block.size-1)
    print('time-info(block): {:.4} ± {:.4}'.format(time_block.mean(), tmp0))


def demo_multiple_stream():
    # totally fail no effects.....
    # TODO CUDA stream http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
    device_cuda = torch.device('cuda')
    device_cpu = torch.device('cpu')
    stream_current = torch.cuda.current_stream()
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()

    for _ in range(10):
        A = torch.empty((100000,), device=device_cuda).normal_(0.0, 1.0)
        with torch.cuda.stream(stream0):
            ret0 = torch.sum(A).item() # sum() may start execution before normal_() finishes!
        # stream_current.wait_stream(stream0)
        ret_ = A.to(device_cpu).numpy().sum().item()
        print(ret_, ret0, abs(ret_-ret0))

    for _ in range(10):
        stream_current.wait_stream(stream0)
        stream_current.wait_stream(stream1)
        with torch.cuda.stream(stream0):
            A = torch.empty((1000000,), device=device_cuda).normal_(0, 1)
        with torch.cuda.stream(stream1):
            ret0 = torch.sum(A)
        with torch.cuda.stream(stream0):
            ret_ = torch.sum(A)
        stream_current.wait_stream(stream0)
        stream_current.wait_stream(stream1)
        ret_ = ret_.item()
        ret0 = ret0.item()
        print(ret0, ret_, ret0-ret_)

    for _ in range(10):
        with torch.cuda.stream(stream0):
            A = torch.empty((1000000,), device=device_cuda).normal_(0, 1)
            tmp0 = torch.sum(A)
        tmp1 = tmp0.item()
        tmp2 = tmp0.item()
        print(tmp1, tmp2, tmp1-tmp2)


# TODO pinned memory

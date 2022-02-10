import utils
import concurrent.futures
import numpy as np
import chainer as ch

def hf_task00():
    ret = 0
    for _ in range(100000):
        utils.THREAD_UNSAFE_FLAG = not utils.THREAD_UNSAFE_FLAG
        ret += (1 if utils.THREAD_UNSAFE_FLAG else -1)
        ret -= (1 if utils.THREAD_UNSAFE_FLAG else -1)
    return ret

def demo_thread_unsafe():
    print('# demo_thread_unsafe')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        job_list = [executor.submit(hf_task00) for _ in range(3)]
        ret = [x.result() for x in job_list]
    print(ret)


def hf_task01():
    ret = 0
    for _ in range(100000):
        ch.config.train = not ch.config.train
        ret += (1 if ch.config.train else -1)
        ret -= (1 if ch.config.train else -1)
    return ret


def demo_thread_safe():
    print('# demo_thread_safe')
    with concurrent.futures.ThreadPoolExecutor() as executor:
        job_list = [executor.submit(hf_task01) for _ in range(3)]
        ret = [x.result() for x in job_list]
    print(ret)


class MyDummyOperator00(ch.FunctionNode):
    def __init__(self, name):
        self.name = name
    def forward(self, inputs):
        return inputs
    def backward(self, target_input_indexes, grad_outputs):
        return grad_outputs
    def __str__(self):
        ret = 'MyDummyOperator00(name={}, id={})'.format(self.name, id(self))
        return ret

class MyDummyModel(ch.Chain):
    def __init__(self):
        super().__init__()
        self.op0 = MyDummyOperator00('op0')
        self.op1 = MyDummyOperator00('op1')
    def __call__(self, x):
        x, = self.op0.apply((x,))
        x, = self.op1.apply((x,))
        return x

class MyDummyFunctionHook(ch.function_hook.FunctionHook):
    def forward_preprocess(self, func, in_data):
        print('[info] hook.forward_preprocess()', func, [x.shape for x in in_data])
    def forward_postprocess(self, func, in_data):
        print('[info] hook.forward_postprocess()', func, [x.shape for x in in_data])
    def backward_preprocess(self, func, in_data, out_grad):
        tmp0 = [(x if x is None else x.shape) for x in in_data]
        tmp1 = [(x if x is None else x.shape) for x in out_grad]
        print('[info] hook.backward_preprocess()', func, tmp0, tmp1)
    def backward_postprocess(self, func, in_data, out_grad):
        tmp0 = [(x if x is None else x.shape) for x in in_data]
        tmp1 = [(x if x is None else x.shape) for x in out_grad]
        print('[info] hook.backward_postprocess()', func, tmp0, tmp1)

def demo_chainer_function_hook():
    print('# demo_chainer_function_hook')
    model = MyDummyModel()
    np0 = np.random.randn(3, 5).astype(np.float32)
    np1 = np.random.randn(3, 5).astype(np.float32)
    ch0 = ch.Variable(np0)
    hook = MyDummyFunctionHook()
    with hook:
        tmp0 = model(ch0)
        tmp0.grad = np1
        tmp0.backward()


if __name__=='__main__':
    demo_thread_unsafe()
    print()
    demo_thread_safe()
    print()
    demo_chainer_function_hook()

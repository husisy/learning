import os
import sys
import time
import subprocess
import threading

def demo_os_pid_ppid():
    print('# demo_os_pid_ppid')
    if os.name!='nt':
        print('[startup] pid={}'.format(os.getpid()))
        child_id = os.fork()
        if child_id==0:
            print('[child] pid:', os.getpid())
            print('[child] ppid:', os.getppid())
        else:
            print('[parent] pid:', os.getpid())
            print('[parent] ppid:', os.getppid()) #usally is the bash/zsh/powershell etc
            print('[parent] child-id: ', child_id)
    else:
        print('os.fork() is not available on windows')


def demo_sys_argv():
    print(sys.argv)
    # python xxx.py  ->  ['xxx.py']
    # python xxx.py 233  ->  ['xxx.py', '233']


def demo_os_pipe_as_stdin():
    stdin_pipe_r, stdin_pipe_w = os.pipe()
    proc = subprocess.Popen(['bash'], stdin=stdin_pipe_r)
    os.write(stdin_pipe_w, b'echo "hello world"\n') #print out hello world
    os.write(stdin_pipe_w, b'exit\n') #become zombie process, see https://stackoverflow.com/q/2760652
    time.sleep(0.01)
    assert proc.poll()==0 #proc stop and delete from /proc
    os.close(stdin_pipe_r)
    os.close(stdin_pipe_w)


def demo_os_pipe_as_stdin_stdout():
    stdin_pipe_r, stdin_pipe_w = os.pipe()
    stdout_pipe_r, stdout_pipe_w = os.pipe()
    proc = subprocess.Popen(['bash'], stdin=stdin_pipe_r, stdout=stdout_pipe_w)
    os.write(stdin_pipe_w, b'echo "hello world"\n')
    tmp0 = os.read(stdout_pipe_r, 1024).decode('utf-8')
    print('[python-print]', tmp0)
    os.write(stdin_pipe_w, b'exit\n')
    time.sleep(0.01)
    assert proc.poll()==0
    os.close(stdin_pipe_r)
    os.close(stdin_pipe_w)
    os.close(stdout_pipe_r)
    os.close(stdout_pipe_w)


def demo_send_eof():
    stdin_pipe_r, stdin_pipe_w = os.pipe()
    proc = subprocess.Popen(['bc'], stdin=stdin_pipe_r)
    os.write(stdin_pipe_w, b'1 + 2\n') #print out 3
    os.close(stdin_pipe_w) #see https://stackoverflow.com/a/13521465
    time.sleep(0.01)
    assert proc.poll()==0 #proc stop and delete from /proc
    os.close(stdin_pipe_r)


class DummyThreadRunner00(threading.Thread):
    def __init__(self, pipe_w):
        super().__init__()
        self.pipe_w = pipe_w
    def run(self):
        for x in range(5):
            tmp0 = str(x)*5
            tmp1 = os.write(self.pipe_w, tmp0.encode('utf-8'))
            print(f'[threading] write {tmp1} bytes:', tmp0)
            time.sleep(1)
        tmp1 = os.write(self.pipe_w, b'$')
        print(f'[threading] write {tmp1} bytes: $')
        os.close(self.pipe_w)


def demo_pipe_as_stream():
    pipe_r, pipe_w = os.pipe()
    worker = DummyThreadRunner00(pipe_w)
    worker.start()
    while True:
        r, _, _ = select.select([pipe_r], [], [])
        assert r[0]==pipe_r
        tmp0 = os.read(pipe_r, 4)
        print(f'[threading] read {len(tmp0)} bytes:', tmp0.decode('utf-8'))
        # no need to time.sleep()
        if '$' in tmp0.decode('utf-8'):
            os.close(pipe_r)
            break


# os.name #'posix' for linux/Unix/MacOS, 'nt' for windows
# os.environ
# os.remove()# 删除文件
# os.rename()# 重命名文件
# os.walk()# 生成目录树下的所有文件名
# os.chdir()# 改变目录
# os.mkdir/makedirs# 创建目录/多层目录
# os.rmdir/removedirs# 删除目录/多层目录
# os.listdir()# 列出指定目录的文件
# os.getcwd()# 取得当前工作目录
# os.chmod()# 改变目录权限
# os.path.basename()# 去掉目录路径，返回文件名
# os.path.dirname()# 去掉文件名，返回目录路径
# os.path.join()# 将分离的各部分组合成一个路径名
# os.path.split()# 返回( dirname(), basename())元组
# os.path.splitext()# 返回 (filename, extension) 元组
# os.path.getatime() .getctime() .getmtime()# 返回最近访问、创建、修改时间
# os.path.getsize()# 返回文件大小
# os.path.exists()# 是否存在
# os.path.isabs()# 是否为绝对路径
# os.path.isdir()# 是否为目录
# os.path.isfile()# 是否为文件

# sys.argv# 命令行参数List，第一个元素是程序本身路径
# sys.modules.keys()# 返回所有已经导入的模块列表
# sys.exc_info()# 获取当前正在处理的异常类,exc_type、exc_value、exc_traceback当前处理的异常详细信息
# sys.exit(n)# 退出程序，正常退出时exit(0)
# sys.hexversion# 获取Python解释程序的版本值，16进制格式如：0x020403F0
# sys.version# 获取Python解释程序的版本信息
# sys.maxint# 最大的Int值
# sys.maxunicode# 最大的Unicode值
# sys.modules# 返回系统导入的模块字段，key是模块名，value是模块
# sys.path# 返回模块的搜索路径，初始化时使用PYTHONPATH环境变量的值
# sys.platform# 返回操作系统平台名称
# sys.stdout# 标准输出
# sys.stdin# 标准输入
# sys.stderr# 错误输出
# sys.exc_clear()# 用来清除当前线程所出现的当前的或最近的错误信息
# sys.exec_prefix# 返回平台独立的python文件安装的位置
# sys.byteorder# 本地字节规则的指示器，big-endian平台的值是'big',little-endian平台的值是'little'
# sys.copyright# 记录python版权相关的东西
# sys.api_version# 解释器的C的API版本

# Standard Library

*注*：STL是cpp语言中Stand Template Library的缩写，为避免歧义，此处使用STDL作为Standard Library的缩写

1. link
   * [documentation](https://docs.python.org/3/library/index.html)

## pdb

1. link
   * [pdb-documentation](https://docs.python.org/3/library/pdb.html)
2. 设置断点
   * `breakpoint()`
   * `import pdb; pdb.set_trace()`
3. REPL模式下执行pdb `pdb.run('hf0')`
4. 命令行下执行pdb `python -m pdb draft00.py`
5. 在pdb模式下的快捷键
   * `h`, `help`: 打印所有快捷键
   * `help xxx`：查询某一快捷键
   * `c`: continue
   * `n`: next line
   * `q`: quit
   * `where`
   * `whatis xxx`
6. `pdb.pm()`: post-mortem debugging验尸。。。。

REPL模式下执行pdb

```python
import pdb
def hf0():
    a = 'hello'
    print(a)
    breakpoint() #seems cannot change the value
    print(a)
pdb.run('hf0()')
```

## unittest

1. link
   * [testing your code](https://docs.python-guide.org/writing/tests/)
2. `python setup.py develop` to do the unittest
3. 偏见
   * 使用`pytest`替代`unittest`
   * do NOT distribute unittest files within the module itself. It often increases complexity for the users and many test suites often require additional dependencies and runtime contexts

## tempfile

1. high level interface: `TemporaryFile() NamedTemporaryFile() TemporaryDirectory() SpooledTemporaryFile()`
   * context manager
2. low level interface: `mkstemp()` `mkdtemp()`
   * cleanup manually
3. `gettempdir() gettempprefix()`
4. recommend to use keyword arguments

## tarfile

1. `gzip`, `bz2`, `lzma`

## pickle

1. link
   * [documentation](https://docs.python.org/3/library/pickle.html)
   * [序列化Python对象](http://python3-cookbook.readthedocs.io/zh_CN/latest/c05/p21_serializing_python_objects.html)
   * [Python pickle 模块学习](http://blog.csdn.net/sxingming/article/details/52164249)
2. 模块`cPickle`已废弃
3. **禁止**对不信任的数据使用`pickle.load()`
4. 特性
   * 将对象以文件的形式存放在磁盘
   * pickle序列化后的数据，可读性差，人一般无法识别
5. 常用示例
   * `with open(filename, 'wb') as fid: pickle.dump(xxx, fid)`, 对象序列化到文件，调用`__getstate__()`获取序列化对象
   * `with open(filename, 'rb') as fid: xxx = pickle.load(fid)`, 文件中恢复对象，调用`__setstate__()`反序列化，反序列对象前要让python能够找到类的定义
   * `s = pickle.dumps(xxx)`, 对象序列化到字符串
   * `xxx = pickle.loads(s)`, 字节流恢复对象
6. `pickle`对于大型的数据结构比如使用`array`或`numpy`模块创建的二进制数组编码效率不高。建议先在一个文件中将其保存为数组数据块或使用更高级的标准编码方式如HDF5 (需要第三方库的支持)
7. 由于`pickle`是Python特有的并且附着在源码上，所有如果需要长期存储数据的时候不应该选用它。 例如，如果源码变动了，你所有的存储数据可能会被破坏并且变得不可读取。 坦白来讲，对于在数据库和存档文件中存储数据时，你最好使用更加标准的数据编码格式如XML，CSV或JSON。 这些编码格式更标准，可以被不同的语言支持，并且也能很好的适应源码变更
8. `pickle.pickler()`
9. `pickler.clear_memo()`

```python
with open(file,'rb') as fid:
    xxx = pickle.load(fid, encoding='bytes')
```

## os and sys

1. link
   * [documentation / os](https://docs.python.org/3/library/os.html)
   * [documentation / sys](https://docs.python.org/3/library/sys.html)
   * [Python中os与sys两模块的区别](http://www.itcast.cn/news/20160831/1848418827.shtml)
2. `os` provides a portable way of using operating system dependent functionality
3. `sys` provides access to some variables used or maintained by the interpreter and to functions that interact strongly with the interpreter

## threading

1. link
   * [莫烦Python-threading](https://morvanzhou.github.io/tutorials/python-basic/threading/1-why/)
   * [python并发4：使用thread处理并发](http://blog.gusibi.com/post/python-thread-note/)
2. 多线程涉及竞争-互斥的问题，在理解之前**不推荐使用**
3. 常用函数`threading.active_count() threading.enumerate() threading.current_thread()`
4. threading不一定有效率：GIL (global interpreter lock)
5. global interpreter lock (GIL)
   * python的确存在GIL，且对于某些运算
   * GIL对文件读取的性能影响很大
   * GIL对numpy性能影响可忽略不计，对`np.fromfile`影响很大
   * numpy因为多线程/多进程同时进行导致的性能降低并没有因为是多线程而降低更多
   * numpy多进程依旧会降低性能，与OpenMP相比如何呢？

## multiprocessing

1. link
   * [documentation](https://docs.python.org/zh-cn/3/library/multiprocessing.html)
   * [莫烦python-multiprocessing](https://morvanzhou.github.io/tutorials/python-basic/multiprocessing/1-why/)
   * [正确使用 Multiprocessing 的姿势](https://jingsam.github.io/2015/12/31/multiprocessing.html)
2. `if __name_=='__main__'`是必须的
3. 进程锁 `multiprocessing.Lock()`
4. 三种启动方法：`spawn fork forserver`
5. 队列Queue是进程线程安全的
6. 进程间共享状态：`multipoocessing.Value multiprocessing.Array`，服务进程`multiprocessing.Manager()`
7. 同步方式：`multiprocessing.Queues`, `multiprocessing.Pipes`, `multiprocessing.Lock`
8. TODO：管理器，代理，自定义管理器

## argparse

1. link
   * [documentation](https://docs.python.org/3/howto/argparse.html)
   * [Argparse 简易教程](https://blog.ixxoo.me/argparse.html)

## logging

1. link
   * [documentation](https://docs.python.org/3/library/logging.html)
2. level `debug info warning error exception critical`, default `warning`

## signal

1. link
   * [documentation](https://docs.python.org/zh-cn/3/library/signal.html)
2. signal**不能**用于线程间通信

## ipaddress

1. link
   * [documentation](https://docs.python.org/3/library/ipaddress.html)
   * [python-howto/introduction](https://docs.python.org/3/howto/ipaddress.html#ipaddress-howto)

## struct

1. link
   * [documentation](https://docs.python.org/zh-cn/3/library/struct.html)
   * [廖雪峰](https://www.liaoxuefeng.com/wiki/1016959663602400/1017685387246080)

## socket

1. link
   * [廖雪峰-网络编程](https://www.liaoxuefeng.com/wiki/1016959663602400/1017787560490144)
   * [Python-doc-socket-programming-HOWTO](https://docs.python.org/3.7/howto/sockets.html)
2. 概念
   * `socket.AF_INET`：IPv4协议
   * `socket.AF_INET6`：IPv6协议
   * `socket.SOCK_STREAM`：面向流的TCP协议
   * `socket.SOCK_DGRAM`：面向无连接的UDP协议
   * Inter Process Communication (IPC)
3. 常见服务端口
   * http服务80
   * https服务443
   * SMTP邮件服务25
   * FTP服务21

## selectors

1. link
   * [stackoverflow](https://stackoverflow.com/questions/53045592/python-non-blocking-sockets-using-selectors)
   * [documentation](https://docs.python.org/3/library/selectors.html)
   * [link0](https://learnku.com/docs/pymotw/selectors-io-multiplexing-abstractions/3428)

## asyncio

1. link
   * [documentation](https://docs.python.org/3/library/asyncio.html)
2. 要求`python37`

## sqlite3

1. link
   * [documentation](https://docs.python.org/3.8/library/sqlite3.html)
2. 对于未知内容的变量，**禁止**用string的方式嵌入到SQL命令中，见[SQL injection attack](https://xkcd.com/327/)
3. `connection`与`cursor`的区别，见[stackoverflow](https://stackoverflow.com/q/6318126/7290857)

## concurrent.futures

1. link
   * [documentation](https://docs.python.org/3/library/concurrent.futures.html)
2. 当且仅当所有进程抛错中断时会导致主进程中断进而打印出错信息（或者进程结束），否则表现为挂起

## hashlib,hmac,secrets

1. link
   * [documentation](https://docs.python.org/3/library/hashlib.html)
   * common weakness enumeration (CWE) [link](https://cwe.mitre.org/index.html)
   * common vulnerabilities and exposures (CVE) [link](https://www.cve.org/)
2. concept
   * secure hash algorithm (message digest algorithm)
   * SHAKE variable length digests
3. file hashing
4. keyed hashing
5. randomized hashing

secrets

1. @2015, `32bytes/256bits` of randomness is sufficient for the typical use (against brute force attack)
2. timing attacks: `secrets.compare_digest`, `hmac.compare_digest`
3. password: salted, hashed

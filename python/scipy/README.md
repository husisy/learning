# scipy

1. link
   * [official site](https://docs.scipy.org/doc/numpy/index.html)
   * [documentation](https://docs.scipy.org/doc/)
   * [user guide](https://docs.scipy.org/doc/numpy/user/) *TODO*
   * [quickstart tutorial](https://numpy.org/devdocs/user/quickstart.html)
   * prerequisite: [The Python Tutorial](https://docs.python.org/3/tutorial/)
   * [Functions and Methods Overview](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html#functions-and-methods-overview)
   * [broadcasting](https://docs.scipy.org/doc/numpy-dev/user/basics.broadcasting.html)
   * [indexing1](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html)
   * [indexing2](https://docs.scipy.org/doc/numpy-dev/user/basics.indexing.html#basics-indexing)
   * [misc00](http://reverland.github.io/python/2012/08/24/scipy/)
   * [misc01](https://segmentfault.com/a/1190000003946953)
2. install
   * `conda install -c conda-forge numpy`
   * `pip install numpy`
3. 优先使用`numpy`，使用`scipy`当且仅当不得不使用`scipy`
4. concept: rank `.ndim`, dimension `.shape`, vectorization, broadcasting
5. broadcasting rule, see [link](https://numpy.org/devdocs/user/basics.broadcasting.html)
   * 最左维度加1
   * 1拓维

| data type | minimum | maximum |
| :-: | :-: | :-: |
| `int16` | `-32768` | `32767` |
| `int32` | `-2147483648` | `2147483647` |
| `int64` | `-9223372036854775808` | `9223372036854775807` |

| data type | sign | exponent | fraction |
| :-: | :-: | :-: | :-: |
| `float16` | `1` | `5` | `10` |
| `float32` | `1` | `8` | `23` |
| `float64` | `1` | `11` | `52` |

TODO

1. 整理[numpy data type objects](https://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html#specifying-and-constructing-data-types)
2. 阅读[numpy reference](https://docs.scipy.org/doc/numpy/reference/)
3. Boolean array indexing
4. [Data type objects](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html)
5. [numpy scalars](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html#id2)
   * same attributes and methods as `ndarrays`; however, immutable, none of the array scalar attributes are settable
   * `isinstance(z1, np.generic)`
   * `isinstance(z1, np.complexfloating)`
6. class `np.ndarray`

## indexing

1. link
   * [python common sequence: indexing rule](https://docs.python.org/3/library/stdtypes.html#common-sequence-operations)
   * [indexing](https://numpy.org/devdocs/user/basics.indexing.html#basics-indexing)
   * [array indexing](https://numpy.org/devdocs/reference/arrays.indexing.html#arrays-indexing)
2. python common sequence: indexing rule
   * 缺省值为`None`
   * `[i:j]`：如`i/j`为负数则替换为`len()+i`或`len()+j`；如`i/j`大于`len()`则替换为`len()`；如`i`缺省则替换为`0`；如`j`缺省则替换为`len()`；indexing范围`i<=?<j`
   * `[i:j:k]`：如`k`缺省则替换为`1`；如`i/j`为负数则替换为`len()+i`或`len()+j`；当`k`为正数时，如`i`缺省则替换为`0`，如`j`缺省则替换为`len()`；当`k`为负数时，如`i`缺省则替换为`len()-1`，如`j`缺省则替换为END VALUE（这一概念在python可编程字符中不存在）
3. `np.newaxis`, `...`
4. abbre: `x[obj]`, `d_0, d_1, ..., d_n`
5. `x[(exp1, exp2, ..., expN)]=x[exp1, exp2, ..., expN]`
6. transversing
   * `for x in np1.flat:`
   * `for ind1, x in np.ndenumerate(np1):`

basic indexing-slicing

1. view
2. `obj`
   * `slice`
   * `interger`
   * `tuple` of `slice` and `integer`
   * backward compatible: `z1[[slice(0,3),1]]`
3. negative indexing: `n_i` means `n_i+d_i`
4. `np.newaxis`, `None`
5. Ellipsis: `...`
6. slice: `start:stop:step`, `i:j:k`
   * `:j:k` $=$ `0:j:k` or `n-1:j:k`
   * `i::k` $=$ `i:n:k` or `i:-n-1:k`
   * `i:j` $=$ `i:j:1`
   * `:` $=$ `::`
   * `0` $\neq$ `0:1`
7. `:` is assumed for any subsequent dimensions

advanced indexing

1. copy
2. `obj`
   * non-tuple sequence object
   * `ndarray` (`integer` or `bool`)
   * tuple with at least one sequence objecct,
3. distinguish
   * `x[(1,2,3)]` = `x[1,2,3]`
   * `x[[1,2,3]]` = `x[(1,2,3),]`
   * `x[[1,2,slice(None)]]` (backward compatiblity required)
4. purely integer array indexing
   * indexing shape
   * indexing broadcasting
   * `np.ix_()`
5. 未理解

```python
# purely integer array indexing
# select corner
# indexing shape, indexing broadcast
z1 = np.array(3,4)
ind1 = np.array([[0], [2]], dtype=np.intp)
ind2 = np.array([[0,3]], dtype=np.intp)
z1[ind1, ind2]

z1 = np.random.rand(3,4)
ind1 = np.array([0,2]).reshape(2,1)
ind2 = np.array([0,3])
z1[ind1,ind2]
```

## sparse

1. sparse matrix type
   * `csc_matrix`: compressed sparse column format
   * `csr_matrix`: compressedd sparse row format，用于matrix vector products
   * `bsr_matrix`: block sparse row format
   * `lil_matrix`: list of lists format，用于创建，支持basic slicing及fancy indexing，方便转化为`csr_matrix`
   * `dok_matrix`: dictionary of keys format，用于创建
   * `coo_matrix`: coordinate format (aka IJV, triplet format)，用于创建
   * `dia_matrix`: diagonal format
2. 对于sparse类型，强烈不建议使用numpy函数
   * 优先使用`scipy.sparse`模块下函数
   * `.toarray()`转化为`np.ndarray`
3. 偏见
   * 使用`.toarray()`而非使用`.todense()`

## Scipy.linalg.interpolative

1. interpolative decomposition (ID)
2. link
   * [documentation - Interpolative matrix decomposition](https://docs.scipy.org/doc/scipy/reference/linalg.interpolative.html)
   * [documentation - ID fortran package](http://tygert.com/id_doc.4.pdf)

## np.ndimage

1. link
   * [official site](https://docs.scipy.org/doc/scipy/reference/tutorial/ndimage.html)
   * [misc00](https://segmentfault.com/a/1190000004002685)
   * [Programming Computer Vision with Python （学习笔记六）](https://segmentfault.com/a/1190000004033826)
2. 高斯滤波器
3. Prewitt滤波器
4. Sobel滤波器

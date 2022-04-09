# MATLAB

```bash
matlab -nodisplay -singleCompThread
```

1. link
   * [official site](https://www.mathworks.com/)
   * [documentation](https://www.mathworks.com/help/matlab/index.html)
   * [get started with MATLAB](https://www.mathworks.com/help/matlab/getting-started-with-matlab.html)
   * [ilovematlab.cn](https://www.ilovematlab.cn/matlab_jishuwenzhang_technical_articles/)
2. MATLAB (MATrix LABoratory)
3. panel
   * current folder: `cd`, `movefile()`, `mkdir()`, `rmdir()`
   * command window: `which`, `clc`, `close`, `format compact`, `KEY-up-arrow`, `KEY-down-arrow`
   * workspace: `ans`, `whos`, `clearvars`, `save()`, `load()`, `clearvars`
   * editor: `edit`, `type`
   * debugger: `debug if error`, `debug if warning`, `dbclear if error`
4. 获得帮助：`help min`, `doc min`
5. 偏见
   * command可以完全被function替代，**当且仅当在交互式命令中使用command**
   * 对于零参数函数，**禁止**使用省略括号的语法糖，即**禁止**将`a=hf0();`写作`a=hf0;`
6. 环境变量PATH: `addpath()`, `savepath()`
7. 绘图
   * `plot()`, `surf()`, `mesh()`
   * `xlabel()`, `title()`, `legend()`, `hold()`
   * `propedit()`

随机数

1. 每次matlab重启都会将随机数种子初始化为0，故重启后的随机数序列是完全相同的
2. `rng('shuffle')`设置的随机数种子与时间相关，时间精度为`0.01s`，即没过`0.01s`对应的随机数种子加1
3. `tic()`返回值的时间精度为`1e-6s`，该精度作为多进程随机数种子应该是足够的 `rng(uint32(bitshift(bitshift(tic(), 32), -32)));`

profile

1. link
   * [documentation/profile-your-code-to-improve-performance](https://www.mathworks.com/help/matlab/matlab_prog/profiling-for-improving-performance.html)
   * [documentation/profile](https://www.mathworks.com/help/matlab/ref/profile.html)
   * [documentation/profile-parallel-code](https://www.mathworks.com/help/parallel-computing/profiling-parallel-code.html)

TODO

1. [function-argument-validation](https://www.mathworks.com/help/releases/R2019b/matlab/matlab_prog/function-argument-validation-1.html)

## data type

1. link
   * [data types](https://www.mathworks.com/help/matlab/data-types_data-types.html)
   * [Fundamental MATLAB Classes](https://www.mathworks.com/help/releases/R2018a/matlab/matlab_prog/fundamental-matlab-classes.html)
2. 基础数据类型
   * scalar: function handle only
   * matrix or array: logical, char (string), numeric (see below), table, cell, struct
   * categorical, datatime, duration, timetable
3. numeric:
   * double (default), single, int8, int16, int32, int64, uint8, uint16, uint32, uint64:
   * cast: `double()`, `single()`, `int8()`, `int16()`, `int32()`, `int64()`, `uint8()`, `uint16()`, `uint32()`, `uint64()`, `cast()`, `typecast()`, `
   * `isinteger()`, `isfloat()`, `isnumeric()`, `isreal()`, `isfinite()`, `isinf()`, `isnan()`
   * `eps()`, `flintmax()`, `Inf()`, `NaN()`, `realmin()`, `realmax()`, `intmin()`, `intmax()`
4. char array, string **seperated documentation TBA**
   * [Characters and Strings](https://www.mathworks.com/help/matlab/characters-and-strings.html)
   * create: `string()`, `strings()`, `join()`
   * `'hello world'`
   * `z1 = ''''`, `z2 = ''''''`
   * `[z1,z2]`
   * `num2str(233)`, `str2num('233')`, `char([65:90,97:122])`, `+'ABCabc'`, `str2double()`
5. complex: **NOT** fundamental class, `1i 1j` (`i j` not recommended)

TODO

1. dates and time
2. categorical arrays
3. tables
4. timetables
5. strut
6. cell
7. function handles
8. map containers
9. time series
10. data type identification
11. data type conversion

## 运算符

1. link
   * [Operators and Elementary Operations](https://www.mathworks.com/help/matlab/operators-and-elementary-operations.html)
   * [MATLAB Operators and Special Characters](https://www.mathworks.com/help/matlab/matlab_prog/matlab-operators-and-special-characters.html)
   * [Operator Precedence](https://www.mathworks.com/help/releases/R2018a/matlab/matlab_prog/operator-precedence.html)
2. 运算符：
   * 算术运算符（所有backslash改写为slash）：`+ - * / ^ '`
   * 关系运算符: `== ~= > >= < <=`
   * 逻辑运算符: `& && | || ~`
3. 特殊运算符
   * `, ; : () [] {}`
   * `.`: decimal point, element-wise operation, struct field access, object property of method specifier
   * `~`: argument placeholder, logical not
   * `@`: function handler, class folder indicator
   * `...`: line continuation
   * `''`: char array
   * `""`: string
   * `!`: operating system command
   * `?`: metaclass for MATLAB class
   * `+`: package directory indicator
4. 运算符与函数对应关系（所有backslash改写为slash）
   * `+  plus()  uplus()`
   * `-  minus()  uminus()`
   * `.*  times()`
   * `.^  power()`
   * `./  rdivide()`
   * `./  ldivide()`
   * `.'  transpose()`
   * `*  mtimes()`
   * `/  mrdivide()`
   * `/  mldivide()`
   * `^  mpower()`
   * `'  ctranspose()`
   * `==  eq()  isequal()  isequaln()`
   * `>  gt()`
   * `>=  ge()`
   * `<  lt()`
   * `<=  le()`
   * `~=  ne()`
   * `~  not()`
   * `&  and()`
   * `|  or()`

## numeric type

1. types
   * `double() single()`
   * `int8() int16() int32() int64() round() fix() floor() ceil()`
   * `uint8() uint16() uint32() uint64()`
   * `eps() Inf() intmax() intmin() NaN() realmax() realmin()`
   * `isfloat() isinteger() isnumeric() isreal() isfinite() isinf() isnan() isa(x,'int32')`

## function

1. link
   * [matlab function list](https://www.mathworks.com/help/matlab/functionlist.html)
   * [command vs function syntax](https://www.mathworks.com/help/matlab/matlab_prog/command-vs-function-syntax.html)
2. misc
   * `disp(rand(4))`
   * `[~,ret2] = max([1,2,3])`
3. `pcode`

## matrix

1. link
   * [matrices and arrays](https://www.mathworks.com/help/releases/R2018a/matlab/matrices-and-arrays.html)
   * [Matrix Indexing](https://www.mathworks.com/help/matlab/math/matrix-indexing.html)
   * [Matrix Indexing in MATLAB](https://www.mathworks.com/company/newsletters/articles/matrix-indexing-in-matlab.html)
   * [set operation](https://www.mathworks.com/help/releases/R2018a/matlab/set-operations.html)
   * [Bit-Wise operations](https://www.mathworks.com/help/releases/R2018a/matlab/bit-wise-operations.html)
2. 创建
   * comma and semicolon: `z1 = [1,2,3;4,5,6]`
   * broadcasting, `repmat()`, `repelem()`, `kron()`
   * `[z1,z1]`, `[z1;z1]`, `cat()`, `horzcat()`, `vertcat()`, `blkdiag()`
   * `ones()`, `zeros()`, `eye()`, `true()`, `false()`, `diag()`
   * `rand()`, `randn()`, `randperm()`
   * `magic()`, `pascal()`
   * `:`, `linspace()`, `logspace()`, `freqspace()`, `meshgrid()`, `ndgrid()`
   * `accumarray()`
3. 高维数组
   * indexed assignment: `z1 = ones(5,5); z1(:,:,2) = zeros(5,5)`
   * function: `rand(4,3,2)`, `repmat()`, `cat()`
4. size and shape
   * `size()`
   * `length()`(not recommand)
   * `ndims()`
   * `numel()`
   * `isscalar()`, `isvector()`, `ismatrix()`, `isrow()`, `iscolumn()`, `isempty()`
   * `reshape()`, `squeeze()`
5. rearrange
   * `sort()`, `sortrows()`, `topkrows()`
   * `issorted()`, `issortedrows()`
   * `flip()`, `fliplr()`, `flipud()`, `rot90()`, `transpose()`, `ctranspose()`, `circshift()`
   * `permute()`, `ipermute()`, `shiftdim()`
6. indexing: `z1 = rand(4,3,2)`
   * `:`, `end`, `ind2sub()`, `sub2ind()`
   * sub-indexing, linear indexing, logical indexing indexing
   * assignment broadcasting
   * `z1(end:-1:1, [3,3,1], :)`
   * `z1(4,2,:)` vs. `z1(4,[2,5])`
   * `z1(z1>0.5)`
   * `z([1,2],1:2)` is same as `z([1;2],1:2)`
7. elements outside an array
   * throw error if reference
   * malloc space to accommodate element if assign (not recommended in loop)
8. delete element: `z1 = rand(4,3)`
   * `z1(:,2) = []`, still matrix
   * `z1(3) = []`, array now
9. preallocating memory
10. misc function
    * `trace()`, `tril()`, `triu()`
    * `bsxfun()`, `arrayfun()`, `cellfun()`
    * `cumprod()`, `cumsum()`, `diff()`, `prod()`, `sum()`
    * `movsum()`, `movmean()`, `movmad()`, `movmax()`, `movmin()`, `movstd()`, `movvar()`, `movprod()`, `movmedian()`
    * `ceil()`, `fix()`, `floor()`, `round()`, `idivide()`, `mod()`, `rem()`
    * `xor()`, `all()`, `any()`
    * `find()`
11. kinds
    * empty matrix: `any(size(z1)==0)`, `size([])==[0,0]`, **zero dimenstion do NOT support broadcasting**
    * scalar: `all(size(z1)==[1,1])`
    * vector: `any(size(z1)==[1,1])`, column vector and row vector
    * matrix: `all(size(z1)>0)`
12. sparse matrix: `full()`, `sparse()`, `speye()`, `sprand()`, `issparse()`, `nnz()`, `nonzeros()`, `nzmax()`, `spalloc()`, `spdiags()`

```MATLAB
A = [];
size(A), length(A), numel(A), any(A), sum(A) %return zero

A = [];
ndims(A), isnumeric(A), isreal(A), isfloat(A), isempty(A), all(A), prod(A) #return an empty array
```

## indexing

1. link
   * [documentation-array indexing](https://www.mathworks.com/help/matlab/math/array-indexing.html)
   * [documentaiont-Access-Data-Cell-Array](https://www.mathworks.com/help/matlab/matlab_prog/access-data-in-a-cell-array.html)
   * [documentation-Access-Data-Using-Categorical-Arrays](https://www.mathworks.com/help/matlab/matlab_prog/access-data-using-categorical-arrays.html)
2. indexing by position, linear indexing, logical indexing
3. indexing by position: `x(float,(4,4))`
   * `x(2,3)`
   * `x(2,[2,3,4])`, `x(2,2:4)`, `x(2,2:end)`
   * `x([1,2],[3,4])`, `x(1:2,3:4)`, `x(1:2,3:end)`
   * `x(1,[1,2,3,4])`, `x(1,:)`
   * support higher-dimensional array
4. linear indexing `x(float,(3,3))`
   * `x(:)`
   * `x(6)`等价于indexing by position`x(3,2)`
   * `sub2ind()`, `ind2sub()`
5. logical indexing
   * `x(x>y)`
   * `is-xxx()`function: `ismissing()`
6. remove element `x(1)=[]`
7. cell array, `x = {'one','two','three'; 1, 2, 3};`
   * cell indexing `x(1,2)`, `x(:,1:2)` (cell array)
   * `cell2mat()`
   * content indexing `x{1,2}`, `x{:,1:2}` (comma-separated list)
   * `[a,b,c] = x{1,:};`, `[x{1,:}]`
8. categorical
   * `== ~=`, `> >= < <=`, `ismember()`, `isundefined()`
   * `summary()`打印所有类别统计信息
   * `removecats()`区别于indexing
   * `categories()`
   * preallocate

## categorical array

1. link
   * [documentation-categorical-array](https://www.mathworks.com/help/matlab/categorical-arrays.html)
2. create
   * 去除首尾空格
   * `categorical({'2','3','3'})`

## table

TODO

## characters and strings

1. link
   * [documentation-characters-and-strings](https://www.mathworks.com/help/matlab/characters-and-strings.html)
2. character array: indexing, concatenate
   * create `'hello world' char() blanks()`
   * escape `''''`
   * to other types `uint16()`
   * concatenate `strcat()`
   * `ischar() isletter() ispace() isstrprop()`
   * `strjust() deblank() strtrim()`
   * expand character array (NOT recommanded)
3. string array
   * create: `"hello world" string() strings()`
   * escape `""""`
   * `strlength()`
   * `contains() ==`
   * `replace() split() join() sort()`

## live scripts

1. 修改m文件后缀名为mlx会破坏文件
2. 运行：ctrl+Enter 或者 点击左侧vertical striped bar
3. output inline buttom
4. clear all output
5. code/text切换：alt+Enter
6. title: # text +Enter
7. heading: ## text+Enter
8. section break with heading: %% text +Enter
9. section break: crl+alt+Enter/%% + Enter
10. bulleted list: */-/+ text
11. numbered list: 1. text
12. LaTeX: \$bula\$

```MATLAB
edit main01.mlx
edit main01 %this will not work
```

## unittest

1. link
   * [知乎-如何使用MATLAB写测试1](https://zhuanlan.zhihu.com/p/20689285)
   * [知乎-如何使用MATLAB写测试3](https://zhuanlan.zhihu.com/p/20699698)

## regex

1. `'c[aeiou]+t'`
2. `[A-Z]`
3. `'\s'`
4. `\w*x\w*`
5. `[Ss]h.`
6. caret: `\^`
7. meta-character: `.`, `[abc$|.*+?-]`, `[^abc$|.*+?-]`, `[a-zA-Z_0-9]`, `\w\W`, `\s\S`, `[ \f\n\r\t\v]`, `\d\D`, `[0-9]`, `\oN\o{N}\xN\x{N}`
8. character representation: `\a\b\f\n\r\t\v`, `\char`, `\\`
9. Quantifiers: `expr*`, `expr?`, `expr+`, `expr{m,n}`, `expr{m,}`

## MATLAB parallel toolbox

1. solution: profile, vectorization, built-in parallel computing support, gpuArray, parfor, parfeval, spmd, tall array, datastore, distributing array, batch
2. delete current pool `delete(gcp('nocreate'))`

parfor

1. parfor-loop iteration are independent (except reduction variable)
2. loop varible must be consecutive increasing integer
3. nested parfor loop limitations:
4. no `return`, no `break`, consider `parfeval`
5. unique classification: loop variable, sliced varible, broadcast varible, reduction varible, temporary varibles
6. no global/persistent variable declarations
7. transparency (only to the direct body, call `save/load` in a function): no `eval clear who whos evlalc evalin assignin(caller) save load(unless assigned to a variable)`
8. cannot call scripts
9. nested parfor loop
   * always run the outer loop in parallel
   * `for ind2 = 1:size(A,2)%invalid`
   * `for ind2 = 1:n%valid`
   * `A(ind1,ind2+1) = ind1+ind2%invalid`
   * `A(ind1,ind2) = ind1+ind2-1%valid`
   * `A(ind1,:) = v; disp(A(ind1,1));%invalid`
   * `A(ind1,:) = v; disp(v(1));%valid`
   * `for ind2 = 1:(n/2)%invalid`
   * `nested function% invalid`
   * `nested function handle + feval%valid`
   * `spmd%invalid`

## MATLAB optimal toolbox

1. `bicg()`: Biconjugate gradient
2. `bicgstab()`: Biconjugate gradient stabilized
3. `bicgstabl()`: Biconjugate gradient stabilized (l)
4. `cgs`: Conjugate gradient squared
5. `gmres`: Generalized minimum residual
6. `lsqr`: Least squares
7. `minres`: Minimum residual
8. `pcg`: Preconditioned conjugate gradient
9. `qmr`: Quasiminimal residual
10. `symmlq`: Symmetric LQ
11. `tfqmr`: Transpose-free quasiminimal residual

## figure

1. link
   * [Graphics Objects](https://www.mathworks.com/help/matlab/creating_plots/graphics-objects.html)
   * [type of plots](https://www.mathworks.com/help/matlab/creating_plots/types-of-matlab-plots.html)
2. 见MATLAB OOP
3. `line charts discrete-data-plots`

![objects hierarchical](https://www.mathworks.com/help/matlab/creating_plots/doccenter_graphicsheirarchy.png)

```MATLAB
tmp1 = {'Position',[1,1,400,400],'Color',[1,1,1]};
hFig = figure(tmp1{:});
hAxes = axes('Parent',hFig);
```

## MATLAB向量化编程

1. cell控制多变量输出：`[x{1:2}] = meshgrid(1:3,4:6)`
2. kron精致的实现`edit kron`
3. 矩阵扩维
   * `a = 0; a(4,4)=0;`
   * `repmat`
   * `a = 1:4;a([1,1,1],:)`
   * `kron`
   * `meshgrid`最多支持三维
   * `ndgrid`
   * 矩阵外积`ones(3,1)*(1:4)`
   * `bsxfun`

```MATLAB
assert()
isequal()
find()
regexp()
polyval([1,1,0],10)
arrayfun()
structfun()
toeplitz()
accumarray
cumprod()
movefun()
sign()
```

## TODO

1. [x] 浏览一遍preference
2. [ ] 浏览一遍matlab function
3. [ ] matlab的窗口界面
4. [ ] matlab常用函数的帮助文档
5. [x] 使用handle来画图
6. [ ] 了解figure axes的所有参数
7. [ ] java和matlab混合编程
8. [ ] file exchange看别人写的代码
9. [ ] 刷cody题目
10. [ ] matlab gui
11. [ ] matlab programming style
12. [x] 熟练使用cell

## matlab performance

1. link
   * [doc/technieues to improve performance](https://www.mathworks.com/help/matlab/matlab_prog/techniques-for-improving-performance.html)
2. 使用function而不是script
3. 使用local function而不是nested function
4. 使用modular programming：减少first-time run costs
5. 预分配空间
6. 矢量化
7. independent operation移至for-loop之外
8. create new variables if data type changes
9. short-circuit operatos
10. 尽量避免全局变量
11. 禁止重载built-ins
12. 禁止data as code，保存为数据文件(csv/txt/mat)更合适
13. 禁止使用`clear all`

TODO 阅读profile相关的文档

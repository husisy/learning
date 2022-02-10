# java

coursera-java程序设计

1. 发展历程
   * 1995 JDK1.0 初创
   * 1998 JDK1.2
   * 2000 JDK1.3 改进
   * 2002 JDK1.4 稳定，assert、logging、java2d、NIO、正则表达式
   * 2004 JDK1.5 语法增加，泛型、foreach、自动装箱拆箱、枚举、可变长参数、静态引入、注记、printf、StringBuilder
   * 2006 JDK1.6 广泛，Compiler API（动态编译）、脚本语言支持、WebService支持
   * (2010 Oracle并购Sun)
   * 2011 JDK1.7 改进：常量、带资源的try、重抛异常
   * 2014 JDK1.8 前进大步
2. Java Community Process (JCP), Java Specification Requests (JSRs)
   * JSR 335: Lambda Expressions
3. 核心机制
   * Java Virtual Machine (JVM)：指令集、寄存器集、类文件结构、堆栈、垃圾收集堆、内存区域
   * 代码安全性检测Code Security
   * Garbage Collection
   * Java Runtime Environment (JRE) = JVM + API (lib)
   * Java Development Kit (JDK) = JRE + Tools
   * JDK-Tools: javac编译器，java执行器，javadoc文档生成器，jara打包器，jdb调试器，appletViewer运行applet程序，javap查看类信息及反汇编，javaw运行图形界面程序
4. Application and applet
   * `appletViewer index.html`
   * applet替代方案：Flash，SilverLight，JavaScript
5. JDK文件目录结构
   * `bin` 可执行工具
   * `jre`
   * `demo` 示例文件
   * `include` 与C相关的头文件
   * `lib` 程序库
   * `db` 数据库相关文件
6. java安装，环境变量`PATH/CLASSPATH/JAVA_HOME`，*TBA*
7. [在线文档](https://docs.oracle.com/javase/8/docs/api/index.html)
8. 数据类型
   * 基本数据类型：整数类型`byte/short/int/long`，浮点类型`float/double`，字符型`char`，布尔型`boolean`，变量在栈
   * 引用数据类型：类`class`，接口`interface`，变量引用在堆
9. 字符使用unicode编码，每个字符占两个字节 `'a'`, `'\u0061'`, `'\n'`
10. 流程结构：顺序，分支，循环
11. 简单语句
    * 方法调用
    * 赋值语句
    * **表达式+分号不是合法语句**
    * 注释：`//`, `/*xxx*/`, `/**xxx*/`
12. 注释文档：`@see @version @author @param @return @exception`

## no reference

1. feature
   * 面向对象
   * 虚拟机运行字节码
   * 标准库
   * 开源社区支持
   * 优点：跨平台，适合于企业和互联网开发、Android移动App开发、大数据应用开发
   * 缺点：语法繁琐，无法操作硬件，GUI效果不佳；不适合底层操作系统开发、桌面应用程序开发、桌面大型游戏开发
2. history
   * SUN公司James Gosling为手持设备开发的嵌入式编程语言
   * 原名Oak，1995年改名为Java
3. 版本
   * Java SE: standard edition
   * Java EE: enterprise edition
   * Java ME: micro edition
4. Java规范
   * Java Specification Request (JSR)
   * Java Community Process (JCP)
   * TCK测试套件
   * RI参考实现
5. Java平台
   * JVM (java virtual machine)
   * JRE (java runtime environment): JVM + Runtime Library
   * JDK (java development toolkit): JRE + 编译器 + 其他开发工具
6. IDE
   * Eclipse：原IBM开发
   * IntelLiJ Idea：JetBrains开发
7. Eclipse
   * workspace / project
   * view / perspective
   * general/refresh
   * general/encoding: UTF-8
   * format documents
   * 插件

面向对象

1. Java规范
   * 类名采用驼峰规范，首字母大写
   * 方法名采用驼峰规范，首字母小写
   * 常量名大写
2. 关键词: `public`, `class`, `static`, `void`, `main`, `true`, `false`
3. special character: `;`, `{}`, `()`, `[]`, `'`, `"`
4. 注释
   * 单行注释：`//`
   * 多行注释：`/*bula*/`
   * 特殊多行注释（写在类与方法定义处，用于自动生成文档）：`/**bula *bula */`
5. 变量
   * `int x = 233;`
   * `String x = "233";`
6. 基本数据类型
   * `long`(64bit), `int`(32bit), `short`(16bit), `byte`(8bit)
   * `double`, `float`
   * `boolean`
   * `char`, `String`
   * `1_234_567`
   * `0x233`, `0b1010`, `233L`
7. 常量`final`：用于替代Magic Number
8. 运算符
   * `+-*/%` 整数的除法结果为整数
   * `++`, `--`
   * `+=`, `-=`
   * 移位`<<`, `>>`, `>>>`
   * 位运算`&|^~`
   * 布尔运算符`>`, `>=`, `<`, `<=`, `==`, `!=`, `!`
   * 短路运算符`&&`, `||`
   * `b?x:y` 分支的类型必须相同
   * 优先级
   * 类型自动提升 与 强制转型`(int)2333L`
9. 浮点数强制转型整数
   * 直接扔掉小数部分
   * 四舍五入技巧`（int) (3.2 + 0.5)`
   * 超出整型范围自动变为最大数
10. 字符串类型
    * 引用类型
    * 转义字符: backslach
    * `null`
11. 数组类型
    * `int[] z1 = new int[5]`; `int[] z1 = {2,23,233,2333,23333}`
    * 初始化为默认值
    * 创建后大小不可改变
    * indexing `z1[0]`
    * `z1.length`
    * 引用类型

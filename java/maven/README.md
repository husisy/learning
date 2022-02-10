# maven tutorials

1. link
   * [official site](https://maven.apache.org/index.html)
   * 廖雪峰-Maven
2. 功能
   * 依赖管理
   * 标准化项目结构
   * 标准化构建流程（编译，打包，发布）
3. 中央仓库：[http://repo1.maven.org/maven2/](http://repo1.maven.org/maven2/)
   * 缓存已下载的jar包`~/.m2/repository`
4. 添加Maven镜像仓库
5. 构建流程
   * `maven clean`: 删除所有编译生成的文件
   * `maven compile`: 编译源码、测试源码，包括validate, initialize, generate-sources, process-sources, process-resources, compile
   * `maven test`: 运行测试
   * `maven package`: 打包为jar/war
6. maven生命周期lifecycle/阶段phase/goal/plugiin
   * clean，对应的plugin是clean
   * compile，对应的goal是compiler:compile；对应的plugin是compiler
   * test，对应的goal是comiler:testCompile, surefile:test；对应的plugin是surefire
   * package，对应的plugin是jar
7. 常用插件
   * maven-shade-plugin：打包所有依赖包并生成可执行jar
   * cobertura-maven-plugin：生成单元测试覆盖率报告
   * findbugs-maven-plugin：对Java源码进行静态分析以找出潜在问题

## pom.xml

1. root: `<project></project>`
2. 唯一标识符
   * `<groupId>com.husisy.misc</groupId>`
   * `<artifactId>test</artifactId>`
   * `<version>0.0.1-SNAPSHOT</version>`
3. output
   * `<packaging>jar</packaging>`
4. `<properties></properties>`
   * `<java.version>1.8</java.version>`
   * `<maven.compiler.source>1.8</maven.compiler.source>`
   * `<maven.compiler.target>1.8</maven.compiler.target>`
5. `<dependencies></dependencies>`
   * `<dependency></dependency>`
6. Maven依赖关系`<dependencies><dependency><scope>xxx</scope></dependency></dependencies>`
   * `compile`: 编译时需要（默认），`commons-logging`
   * `test`: 编译Test是需要，`junit`
   * `runtime`: 编译是不需要，但运行时需要，`log4j`
   * `provided`: 编译时需要用到，但运行时由JDK或某个服务器提供，`servlet-api`

```Java
/*
a-maven-project
+- src
|  +- main
|     +- java //java源码
|     +- resources //资源文件
|  +- test
|     +- java //java测试源码
|     +- resources //测试资源文件
+- target //编译输出
+- pom.xml // 项目描述文件
*/
```

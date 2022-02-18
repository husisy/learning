# bazel

## setup workspace

1. link
   * [bazel documentation](https://docs.bazel.build/versions/master/tutorial/cpp.html)

```bash
mkdir ws01
cd ws01
git clone https://github.com/bazelbuild/examples/
rm -rf examples/.git
```

## cplusplus

1. bazel build at path `/examples/cpp-tutorial/stage1`
   * win/powershell (below use this only): `bazel build //main:hello-world`
   * win/git (not recommand): `bazel build ///main:hello-world`
2. run: `bazel-bin/main/hello-world`
3. dependency graph: `bazel query --nohost_deps --noimplicit_deps 'deps(//main:hello-world)' --output graph`
   * `sudo apt-get install graphviz xdot`
   * `xdot <(bazel query --nohost_deps --noimplicit_deps 'deps(//main:hello-world)' --output graph)`
   * win: [WebGraphviz](http://www.webgraphviz.com/)

## java

`bazel build //src/main/java/com/example/cmdline:runner_deploy.jar`

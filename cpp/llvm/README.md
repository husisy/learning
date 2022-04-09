# LLVM

1. link
   * official site
   * [github/llvm-ir-tutorial](https://github.com/Evian-Zhang/llvm-ir-tutorial)
   * [tutorial](https://llvm.org/docs/tutorial/index.html)
2. `sudo apt install clang-10 llvm-10`
3. 预处理、语法分析、语义分析
4. abstract struct tree (AST)
5. LLVM IR
   * 内存中的LLVM IR
   * 比特码形式的LLVM IR: `clang -c -emit-llvm draft00.c`
   * 可读形式的LLVM IR: `clang -S -emit-llvm draft00.c`
6. 编译过程
   * `draft00.c`
   * frontend: AST
   * frontend: LLVM IR
   * opt: LLVM IR
   * llc: `.s` Assembly
   * OS Assembly: `.o`
   * OS linker: executable
7. bash last exit code `$?`
8. concept
   * data layout
   * target triple
   * 注释 `;`
   * 所有的全局变量的名称都需要用`@`开头
   * called-saved register, calling-saved register
   * 全局变量和栈上变量皆指针
   * Static Single Assignment (SSA)
9. 查看符号表`nm draft00.exe`
   * `.syntab`, `.dynsym`
   * 控制符号表：链接[linkage-type](http://llvm.org/docs/LangRef.html#id1217)和可见性[visibility-styles](http://llvm.org/docs/LangRef.html#id1219)
10. 当不需要操作地址并且寄存器数量足够时，可以直接使用寄存器，且LLVM IR的策略保证了我们可以使用无数的虚拟寄存器

```c
// draft00.c
int main() {
   return 0;
}
```

```bash
clang-10 draft00.c -o draft00.exe
clang-10 -Xclang -ast-dump -fsyntax-only draft00.c
clang-10 -S -emit-llvm draft00.c #draft00.ll
# opt-10 draft00.ll -S --O3
clang-10 -c -emit-llvm draft00.c #draft00.bc
clang-10 -S -emit-llvm -O3 draft00.c #draft00.ll
llc-10 draft00.ll #draft00.s

clang-10 -S -emit-llvm draft00.c #c -> ll
clang-10 -c -emit-llvm draft00.c #c -> bc
llvm-as draft00.ll #ll -> bc
llvm-dis draft00.bc #bc -> ll
clang-10 draft00.ll -o draft00.exe #ll -> exe
clang-10 -S draft00.c #c -> s

nm draft00.exe
```

```c
int max(int a, int b) {
    if (a > b) {
        return a;
    } else {
        return b;
    }
}

int main() {
    int a = max(1, 2);
    return 0;
}
```

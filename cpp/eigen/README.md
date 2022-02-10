# Eigen

1. link
   * [official site](http://eigen.tuxfamily.org/)
   * [bitbucket](https://bitbucket.org/eigen/eigen/)
   * [documentation](http://eigen.tuxfamily.org/dox/)
2. minimum working example: `g++ draft00.cpp -o tbd00.exe -I C:/Users/zchao/Documents/cplusplus/STL/eigen`
3. [configure vscode](https://code.visualstudio.com/docs/cpp/config-mingw)
   * build task: `ctrl+shift+b`
   * auto format: `alt+shift+f`
   * debug **TODO**
4. `Matrix<Scalar, Rows, Cols>`
   * `Scalar`: `float, double`
   * `typedef Matrix<float,4,4> Matrix4f`
   * `typedef Matrix<float, 3, 1> Vector3f`
   * `typedef Matrix<int, 1, 2> RowVector2i`
   * `typedef Matrix<double,Dynamic,Dynamic> MatrixXd`
   * `typedef Matrix<int,Dynamic,1> VectorXi`
5. column major

## minimum working example

1. 从官网下载的Eigen文件 `3.3.7.zip`
2. 从zip文件中将`./eigen-eigen-323c052e1731/Eigen`文件夹复制至`SOMEWHERE/STL`目录下
3. 在工作目录下编写`draft.cpp`，见下
4. 在工作目录下编译 `g++ draft.cpp -o tbd00.exe -I SOMEWHERE/STL`
5. 在工作目录下运行 `./tbd00.exe`
6. use fixed-size matrices for size 4-by-4 and smaller `Eigen::Matrix3d`

```cpp
#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;
int main()
{
    MatrixXd m(2, 1);
    m(0, 0) = 3;
    m(0, 0) = m(1, 0) + 2.5;
    std::cout << m << std::endl;
}
```

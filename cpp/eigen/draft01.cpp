#include <iostream>
#include <Eigen/Dense>

using Eigen::Matrix3d, Eigen::Vector3d;
using Eigen::MatrixXd, Eigen::VectorXd;

void test00()
{
    std::cout << "\n#test00\n";

    MatrixXd z0 = MatrixXd::Random(3, 3);
    VectorXd z1(3);
    z1 << 1, 2, 3;

    std::cout << "random matrixXd:\n"
              << z0 << std::endl;
    std::cout << "vectior initialized from comma expression:\n"
              << z1 << std::endl;

    z0 = z0 * 2 + MatrixXd::Constant(3, 3, 1.2);
    std::cout << "add some constant:\n"
              << z0 << std::endl;
    std::cout << "matrix*vector:\n"
              << z0 * z1 << std::endl;
}

void test01()
{
    std::cout << "\n#test01\n";

    Matrix3d z0 = Matrix3d::Random();
    Vector3d z1(1, 2, 3);

    std::cout << "random matrix3d:\n"
              << z0 << std::endl;
    std::cout << "vectior3d:\n"
              << z1 << std::endl;

    z0 = z0 * 2 + Matrix3d::Constant(3, 3, 1.2);
    std::cout << "add some constant:\n"
              << z0 << std::endl;
    std::cout << "matrix*vector:\n"
              << z0 * z1 << std::endl;
}

//g++ draft01.cpp -o tbd00.exe -I C:/Users/zchao/Documents/cplusplus/STL -std=c++17
int main()
{
    test00();
    test01();

    MatrixXd z0(2,2);
    z0 << 1, 2,
        3, 4;
    std::cout << z0(0) << "," << z0(1) << "," << z0(2) << "," << z0(3) << std::endl;;
    return 0;
}
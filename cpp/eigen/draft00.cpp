#include <iostream>
#include <Eigen/Dense>
using Eigen::MatrixXd;

// g++ draft00.cpp -o tbd00.exe -I C:/Users/zchao/Documents/cplusplus/STL
int main()
{
    MatrixXd m(2, 1);
    m(0, 0) = 3;
    m(1, 0) = m(0, 0) + 2.5;
    std::cout << m << std::endl;
}

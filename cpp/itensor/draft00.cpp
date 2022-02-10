#include <iostream>
#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <complex>
// using namespace itensor;
using namespace std::complex_literals;

// compile comand see README.md
// itensor::prime
// itensor::primeLevel
// itensor::noPrime
// itensor::order
// Print
// PrintData
// itensor::norm
// itensor::randomITensor
// itensor::svd
// itensor::sqr
// itensor::Matrix
// itensor::QN
// itensor::Out
// itensor::hasQNs

void demo_basic00()
{
    std::cout << "\n[info] demo_basic00" << std::endl;
    auto i = itensor::Index(2, "index i");
    auto j = itensor::Index(3, "index j");
    auto T = itensor::ITensor(i, j);

    T.set(i=1, j=1, 3.14159);
    T.set(i=1, j=2, 2.0+0.3i);
    T.set(i=1, j=3, 2.0+0.33i);

    // Print(T);
    PrintData(T); //itensor::PrintData()
    std::cout << "T: " << T << std::endl;
    std::cout << "T[i=1,j=1]: " << itensor::eltC(T, i=1,j=1) << std::endl;
    std::cout << "T[i=1,j=2]: " << itensor::eltC(T, i=1,j=2) << std::endl;
}


void demo_index00()
{
    std::cout << "\n[info] demo_index00" << std::endl;
    auto i0 = itensor::Index(4);
    std::cout << "Index(i0): " << i0 << std::endl;
    std::cout << "dim(Index): " << itensor::dim(i0) << std::endl;
    auto i1 = i0; //make a copy
    std::cout << "equal(copy): " << (i1==i0) << std::endl;
    std::cout << "default primeLevel(i0): " << itensor::primeLevel(i0) << std::endl;
    // i0.prime(2);
    // i0.noPrime();
    // itensor::prime;
    // itensor::hasTags
    // itensor::tags
    // itensor::addTags
    // itensor::removeTags
    auto I = itensor::Index(itensor::QN(0), 1, itensor::QN(1), 1, itensor::Out, "I");
    std::cout << "QN index:\n" << I << std::endl;

    // auto i = itensor::Index(3, "index i,Link");
}


void demo_matrix()
{
    std::cout << "\n[info] demo_matrix" << std::endl;
    auto M = itensor::Matrix(2,2);
    std::cout << "itensor::Matrix-M:\n" << M << std::endl;
}

void test00()
{
    std::cout << "\n[info] test00" << std::endl;
    auto i = itensor::Index(3);

    auto T = itensor::ITensor(i);
    auto iva = (i=2);
    T.set(iva, 2.33);
    std::cout << "T[iva]: " << itensor::elt(T, iva) << std::endl;
}

void demo_serilazation()
{
    // auto j = itensor::Index(5,"Link");
    // auto k = itensor::Index(2,"Site");
    // auto S = itensor::randomITensor(j,k);
    // itensor::writeToFile("tbd00.it", S); //binary format
}


void demo_siteset()
{
    std::cout << "\n[info] demo_siteset" << std::endl;
    auto sites = itensor::SpinHalf(100);
    auto l = itensor::Index(5,"left");
    auto r = itensor::Index(4,"right");
    auto T = itensor::ITensor(l,sites(3),r);
    auto up5 = sites(5, "Up"); // SpinHalf{Up Dn} Electron{Emp Up Dn UpDn}
    std::cout << "indexVal: " << up5 << std::endl;
}

int main()
{
    demo_basic00();
    demo_index00();
    demo_matrix();
    test00();
    demo_siteset();
    return 0;
}

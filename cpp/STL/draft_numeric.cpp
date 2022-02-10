#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <functional>

void test_accumulate()
{
    // https://en.cppreference.com/w/cpp/algorithm/accumulate
    std::cout << "\n# test_accumulate\n";
    std::vector<int> z0{1,2,3,4,5,6,7,8,9,10};
    int z1[] = {1,2,3,4,5,6,7,8,9,10};
    const int *z2_begin = z1;
    const int *z2_end = z1 + (sizeof(z1)/sizeof(z1[0]));

    int sum0 = std::accumulate(z0.begin(), z0.end(), 0);
    int sum1 = std::accumulate(std::begin(z1), std::end(z1), 0);
    int sum2 = std::accumulate(z2_begin, z2_end, 0); //strange
    int product0 = std::accumulate(z0.begin(), z0.end(), 1, std::multiplies<int>());

    auto hf_dash_fold = [](std::string a, int b) {return std::move(a) + "-" + std::to_string(b);};
    std::string fold_string = std::accumulate(std::next(z0.begin()), z0.end(), std::to_string(z0.front()), hf_dash_fold);

    std::cout << "assumulate(vector<int>): " << sum0 << std::endl;
    std::cout << "assumulate(int[]): " << sum1 << std::endl;
    std::cout << "assumulate(int*): " << sum2 << std::endl;
    std::cout << "assumulate(vector<int>, std::multiplier): " << product0 << std::endl;
    std::cout << "accumulate(vector<int>, hf_dash_fold): " << fold_string << std::endl;
}

void test_inner_product()
{
    std::cout << "\n# test_inner_product\n";

    std::vector<int> x0 {2,3,3};
    int x1[] = {2,23,233};
    int ret_ = 0;
    for (int i=0; i<x0.size(); i++)
    {
        ret_ += x0[i] * x1[i];
    }
    int ret0 = std::inner_product(x0.cbegin(), x0.cend(), std::cbegin(x1), 0);
    std::cout << "for-loop inner_product: " << ret_ << std::endl;
    std::cout << "std::innner_product: " << ret0 << std::endl;
}

// g++ draft_numeric.cpp -std=c++17 -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << "draft_numeric.cpp\n";

    test_accumulate();
    test_inner_product();

    return 0;
}

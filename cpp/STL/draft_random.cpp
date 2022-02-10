#include <iostream>
#include <string>
#include <map>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <ctime>

#include "zcutils.cpp"

// random_device -> random_generator -> random_distribution
// default_generator -> random_distribution

std::mt19937_64* la_ji_cpp_r_generator()
{
    static bool first_entry = true;
    static std::mt19937_64 *r_generator;
    if (first_entry)
    {
        std::random_device r_device;// Seed with a real random value, if available
        std::seed_seq seed{
                r_device()+(unsigned int)(time(nullptr)),
                r_device()+(unsigned int)(time(nullptr)),
                r_device()+(unsigned int)(time(nullptr)),
                r_device()+(unsigned int)(time(nullptr)),
                r_device()+(unsigned int)(time(nullptr)),
                r_device()+(unsigned int)(time(nullptr)),
        };
        // r_device is prefered over std::default_random_engine, see https://en.cppreference.com/w/cpp/header/random
        // time(nullptr) is required, otherwise same results once compiled
        // six copy of r_device... no idea, but it's quite often saw
        r_generator = new std::mt19937_64(seed);
        first_entry = false;
    }
    return r_generator;
}

std::vector<int> random_integer(int lower, int upper_exclude, int num_sample)
{
    assert(upper_exclude > lower);
    std::vector<int> ret(num_sample, 0);
    std::mt19937_64 *r_generator = la_ji_cpp_r_generator();
    std::uniform_int_distribution<int> z0(lower, upper_exclude-1);
    for (int ind0 = 0; ind0 < num_sample; ind0++)
    {
        ret[ind0] = z0(*r_generator);
    }
    return ret;
}

int random_integer(int lower, int upper_exclude)
{
    std::mt19937_64 *r_generator = la_ji_cpp_r_generator();
    return std::uniform_int_distribution<int>(lower, upper_exclude-1)(*r_generator);
}

void test_random_uniform_integer()
{
    std::cout << "\n# test_random_uniform_integer\n";

    int num_sample = 10000;

    std::cout << "## test random_integer(a,b,N)\n";
    std::vector<int> r_sample0(random_integer(0, 4, num_sample));
    std::map<int, int> histgram0;
    for (int &x: r_sample0)
    {
        ++histgram0[x];
    }
    std::cout << "num_sample: " << num_sample << std::endl;
    for (auto x: histgram0)
    {
        std::cout << "count(" << x.first << "): " << x.second << std::endl;
    }

    std::cout << "## test random_integer(a,b)\n";
    std::vector<int> r_sample1;
    std::map<int, int> histgram1;
    for (int ind0=0; ind0<num_sample; ind0++){
        r_sample1.push_back(random_integer(0, 4));
        ++histgram1[r_sample1[ind0]];
    }
    std::cout << "num_sample: " << num_sample << std::endl;
    for (auto x: histgram1)
    {
        std::cout << "count(" << x.first << "): " << x.second << std::endl;
    }
}

std::vector<double> random_uniform(double lower, double upper, int num_sample)
{
    assert(upper >= lower);
    std::vector<double> ret(num_sample, 0);
    std::mt19937_64 *r_generator = la_ji_cpp_r_generator();
    std::uniform_real_distribution<double> z0(lower, upper);
    for (int ind0 = 0; ind0 < num_sample; ind0++)
    {
        ret[ind0] = z0(*r_generator);
    }
    return ret;
}

double random_uniform(double lower, double upper)
{
    std::mt19937_64 *r_generator = la_ji_cpp_r_generator();
    return std::uniform_real_distribution<double>(lower, upper)(*r_generator);
}

void test_random_uniform_real()
{
    std::cout << "\n# test_random_uniform_real(0,2)\n";

    int num_sample = 10000;
    std::vector<double> r_sample(random_uniform(0, 2, num_sample));

    double min = *std::min_element(r_sample.rbegin(), r_sample.rend());
    double max = *std::max_element(r_sample.rbegin(), r_sample.rend());
    double mean = std::accumulate(r_sample.rbegin(), r_sample.rend(), 0.0) / num_sample;
    auto hf0 = [](double a, double b){return a + b*b;};
    double second_moment = std::accumulate(r_sample.rbegin(), r_sample.rend(), 0.0, hf0) / num_sample;
    double variance = second_moment - mean*mean;
    std::cout << "min: " << min << std::endl;
    std::cout << "max: " << max << std::endl;
    std::cout << "mean: " << mean << std::endl;
    std::cout << "second_moment: " << second_moment << std::endl;
    std::cout << "variance: " << variance << std::endl;
}

std::vector<int> random_permutation(int upper_exclude, int num)
{
    assert((upper_exclude>0) && (num>0) && (num<=upper_exclude));
    auto z0 = range(upper_exclude);
    std::mt19937_64 *r_generator = la_ji_cpp_r_generator();
    std::shuffle(z0.begin(), z0.end(), *r_generator);
    return std::vector<int>(z0.begin(), z0.begin()+num);
}

void test_random_permutation()
{
    std::cout << "\n# test_random_uniform_integer\n";

    std::cout << "random_permutation(10,10): " << random_permutation(10,10) << std::endl;
    std::cout << "random_permutation(10,10): " << random_permutation(10,10) << std::endl;
    std::cout << "random_permutation(10,5): " << random_permutation(10,5) << std::endl;
    std::cout << "random_permutation(10,1): " << random_permutation(10,1) << std::endl;
}


// g++ draft_random.cpp -std=c++11 -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << "draft_numeric.cpp\n";

    test_random_uniform_real();
    test_random_uniform_integer();
    test_random_permutation();

    return 0;
}

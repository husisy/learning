#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <ctime>

void test_tic_toc_in_second()
{
    std::cout << "\n# test_tic_toc_in_second\n";
    std::time_t time_start(std::time(nullptr));
    std::this_thread::sleep_for(std::chrono::milliseconds(233));
    std::cout << "elapsed " << std::difftime(std::time(nullptr), time_start) << " seconds\n";
}

void test_tic_toc_in_nanosecond()
{
    std::cout << "\n# test_tic_toc_in_nanosecond\n";
    auto t_start = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(233));
    auto t_end = std::chrono::high_resolution_clock::now();
    double t_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count() / 1e9;
    std::cout << "elapsed " << t_duration << " seconds\n";
}


// g++ draft_ctime.cpp -std=c++11 -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << "draft_ctime.cpp\n";

    test_tic_toc_in_second();
    test_tic_toc_in_nanosecond();

    return 0;
}

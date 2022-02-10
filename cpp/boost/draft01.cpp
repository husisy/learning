#include <iostream>
#include <string>

#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

void test_boost_format()
{
    std::cout << "\n# test_boost_format\n";
    boost::format tmp0("%1% %2%");
    tmp0 % "hello";
    tmp0 % std::string("world");
    std::cout << "format()%(string): " << tmp0 << std::endl;

    auto tmp1 = boost::format("%1% + %2% = %3%") % 22 % 33 % (22 + 33);
    std::cout << "format()%(int): " << tmp1 << std::endl;

    std::string tmp2 = (boost::format("%2% %1%") % "hello" % "world").str();
    std::cout << "format->string: " << tmp2 << std::endl;

    std::cout << "[zc-info]: " << (boost::format("%1%")%1e-2) << std::endl;
    std::cout << "[zc-info]: " << (boost::format("%1%")%1e-5) << std::endl;
    std::cout << "[zc-info]: " << (boost::format("%1%")%1e-9) << std::endl;
    std::cout << "[zc-info]: " << (boost::format("%1%")%1e-15) << std::endl;
    std::cout << "[zc-info]: " << (boost::format("%1%")%1e-30) << std::endl;
    std::cout << "[zc-info]: " << (boost::format("%1%")%1e-100) << std::endl;
    std::cout << "[zc-info]: " << (boost::format("%1%")%0.0) << std::endl;
}

void test_boost_algorith_string()
{
    std::cout << "\n# test_boost_algorith_string\n";
    std::string tmp0("hello world");
    std::cout << "to_upper_copy(): " << boost::algorithm::to_upper_copy(tmp0) << std::endl;
    std::cout << "after to_upper_copy(): " << tmp0 << std::endl;
    boost::algorithm::to_upper(tmp0);
    std::cout << "to_upper(): " << tmp0 << std::endl;

    std::string tmp1("\t hello world\n");
    std::cout << "trim_copy(): %" << boost::algorithm::trim_copy(tmp1) << "%" << std::endl;
    std::cout << "after trim_copy(): %" << tmp1 << "%" << std::endl;
    boost::algorithm::trim(tmp1);
    std::cout << "trim: %" << tmp1 << "%\n";

    std::cout << "starts_with(abc,ab): " << boost::algorithm::starts_with("abc", "ab") << std::endl;
    std::cout << "starts_with(abc,b): " << boost::algorithm::starts_with("abc", "b") << std::endl;
    std::cout << "starts_with(abc,Ab): " << boost::algorithm::starts_with("abc", "Ab") << std::endl;
    std::cout << "istarts_with(abc,Ab): " << boost::algorithm::istarts_with("abc", "Ab") << std::endl;
    // boost::algorithm::ends_with
    // boost::algorithm::contains
    // boost::algorithm::trim_left

    std::string tmp2("abb233");
    std::cout << "trim_left_if(abb233, abc): " << boost::algorithm::trim_left_copy_if(tmp2, boost::algorithm::is_any_of("abc")) << std::endl;
}

void test_boost_algorithm_string_classification()
{
    std::cout << "\n# test_boost_string_classification\n";
    auto tmp0 = boost::algorithm::is_lower();
    auto hf0 = [&tmp0](std::string x) { return boost::algorithm::all(x, tmp0); };
    std::cout << "all(hello,is_lower()): " << hf0("hello") << std::endl;
    std::cout << "all(hEllo,is_lower()): " << hf0("hEllo") << std::endl;

    auto tmp1 = boost::algorithm::is_from_range('a', 'g') || boost::algorithm::is_digit();
    auto hf1 = [&tmp1](std::string x) { return boost::algorithm::all(x, tmp1); };
    std::cout << "all(a233, [a-g]|digit)" << hf1("a233") << std::endl;
    std::cout << "all(h233, [a-g]|digit)" << hf1("h233") << std::endl;

    auto tmp2 = boost::algorithm::is_any_of("helo");
    auto hf2 = [&tmp2](std::string x) { return boost::algorithm::all(x, tmp2); };
    std::cout << "all(hello, is_any_of(helo)): " << hf2("hello") << std::endl;
    std::cout << "all(world, is_any_of(helo)): " << hf2("world") << std::endl;
}

// g++ draft01.cpp -std=c++11 -o tbd00.exe -I $env:CPP_THIRD_PARTY_INCLUDE
int main(int argc, char *argv[])
{
    std::cout << "# draft01.cpp" << std::endl;
    test_boost_format();
    // test_boost_algorith_string();
    // test_boost_algorithm_string_classification();

    //TODO https://www.boost.org/doc/libs/1_71_0/doc/html/string_algo/usage.html
    // std::string tmp0("hello dolly");
    // auto ind_range = boost::algorithm::find_last(tmp0, "ll");
    // auto hf0 = [](char x){return (char)(x+1);};
    // std::transform(ind_range.begin(), ind_range.end(), hf0);
    // std::cout << "test: " << tmp0 << std::endl;
    // boost::algorithm::replace_range(tmp0, ind_range,)
    return 0;
}

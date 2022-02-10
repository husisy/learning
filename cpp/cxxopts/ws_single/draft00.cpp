#include <iostream>
#include <string>

#include <cxxopts.hpp>

// std::string argparse_string(int, argc, char *argv[], )

bool parse_bool_example(int argc, char *argv[], std::string key, bool default_value=false)
{
    // if "true", then true
    // if "false", then false
    // if None, then default_value
    // otherwise, default_value

    std::string escape_string("false");
    cxxopts::Options options("name", "discription");
    options.add_options()(key, "test bool", cxxopts::value<std::string>());
    auto arg_parse = options.parse(argc, argv);
    if (arg_parse.count(key))
    {
        std::string tmp0 = arg_parse["bool"].as<std::string>();
        if (tmp0=="true")
        {
            return true;
        }else if (tmp0=="false")
        {
            return false;
        }else
        {
            return default_value;
        }
    }else
    {
        return default_value;
    }
}

void test_bool_example()
{

}

// g++ draft00.cpp -std=c++11 -o tbd00.exe -I include
int main(int argc, char *argv[])
{
    bool tmp0 = parse_bool_example(argc, argv, "bool", false);
    if (tmp0)
    {
        std::cout << "bool is true\n";
    }else
    {
        std::cout << "bool is false\n";
    }
    // cxxopts::Options options("test_program", "one line description for the test_program");
    // auto options_adder = options.add_options();
    // options_adder("d,debug", "enable debug", cxxopts::value<std::string>()->default_value("false"));
    // // options_adder("f,file", "filename", cxxopts::value<std::string>());
    // auto arg_parse = options.parse(argc, argv);

    // std::cout << "debug: " << arg_parse["debug"].as<std::string>() << std::endl;
    // if (arg_parse["debug"].as<std::string>()!="false")
    // {
    //     std::cout << "la ji c++" << std::endl;
    // }

    return 0;
}

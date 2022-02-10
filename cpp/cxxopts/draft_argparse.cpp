#include <iostream>
#include <string>
#include <map>
#include <cassert>


std::map<std::string,std::string> argparse(int argc, char *argv[])
{
    std::map<std::string, std::string> ret;

    // default value
    ret["para_bool"] = "true";
    ret["para_int"] = "234";
    ret["para_double"] = "-2.34";
    ret["para_string"] = "abb";

    int ind0 = 1; //ignore self
    while ((ind0+1) < argc)
    {
        std::string key(argv[ind0]);
        ind0 = ind0 + 1;
        if (!((key.size()>2) && (key[0]=='-') && (key[1]=='-'))){
            continue;
        }
        key = key.substr(2, key.size()-2);
        if (ret.find(key)==ret.end()){
            continue;
        }
        std::string value(argv[ind0]);
        ret[key] = value;
    }
    return ret;
}

// g++ draft_argparse.cpp -std=c++11 -o tbd00.exe -I include
// ./tbd00.exe
// ./tbd00.exe --para_int 233 --para_string la_ji_cpp --para_bool false --para_double 2.33
// ./tbd00.exe noise0 --para_int 233 noise1 --para_string la_ji_cpp noise2
int main(int argc, char *argv[])
{
    std::map<std::string,std::string> kwargs = argparse(argc, argv);

    bool para_bool = (kwargs["para_bool"]=="true")?true:false;
    int para_int = std::stoi(kwargs["para_int"]);
    double para_double = std::stod(kwargs["para_double"]);
    std::string para_string = kwargs["para_string"];

    std::cout << "para_bool: " << para_bool << std::endl;
    std::cout << "para_int: " << para_int << std::endl;
    std::cout << "para_double: " << para_double << std::endl;
    std::cout << "para_string: " << para_string << std::endl;
    return 0;
}

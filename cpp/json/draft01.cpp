#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>

#include <nlohmann/json.hpp>

void test_constructor()
{
    std::cout << "\n# test_constructor\n";
    nlohmann::json z0;
    z0["pi"] = 3.14;
    z0["name"]["first"] = "nlohmann";
    z0["name"]["last"] = "json";
    z0["list"] = {2,23,233};
    // z0["list"] = std::vector<int> {2,23,233};

    std::cout << "## json[key]=value: " << z0 << std::endl;
    std::cout << ".size(): " << z0.size() << std::endl;

    nlohmann::json z1{{"pi", 3.14}, {"name", {"first", "nlohmann"}, {"last", "json"}}, {"list", {2,23,233}}};
    std::cout << "## json (key,value) list: " << z1 << std::endl;
}

void test_serialization()
{
    std::cout << "\n# test_serialization\n";

    nlohmann::json z0 = nlohmann::json::parse("{ \"happy\": true, \"pi\": 3.14 }");
    std::string z1 = z0.dump();
    std::cout << "json object: " << z0 << std::endl;
    std::cout << "string object: " << z1 << std::endl;

    std::ofstream fout("tbd00/tbd00.json");
    fout << z0;
    fout.close();
    std::ifstream fin("tbd00/tbd00.json");
    nlohmann::json z2;
    fin >> z2;
    fin.close();
    std::cout << "origin json object: " << z0 << std::endl;
    std::cout << "write/read from file: " << z2 << std::endl;
}

void test_misc00()
{
    std::cout << "\n# test_serialization\n";

    nlohmann::json z0{{"0",0}, {"1", 1}, {"3",3}};
    std::string ret("0");
    for (auto &x: z0.items())
    {
        int tmp0 = std::stoi(x.key());
        if (tmp0>=std::stoi(ret))
        {
            ret = std::to_string(tmp0 + 1);
        }
    }
    std::cout << "ret: " << ret << std::endl;
}

// g++ draft01.cpp -std=c++11 -o tbd00.exe -I $env:CPP_THIRD_PARTY_INCLUDE
int main(int argc, char const *argv[])
{
    std::cout << "# draft01.cpp\n";
    test_constructor();
    test_serialization();

    nlohmann::json z0,z1;
    z1["world"] = "233";
    z0["hello"] = z1;
    std::cout << "json as value: " << z0 << std::endl;

    std::cout << "fansile: " << std::to_string(233) << std::endl;
    test_misc00();
    return 0;
}

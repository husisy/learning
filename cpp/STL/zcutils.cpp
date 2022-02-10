#include <vector>
#include <iostream>
#include <cassert>

template<typename T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& t)
{
    s << "[";
    for (const auto &x: t)
    {
        s << x << ",";
    }
    s << "]";
    return s;
}

std::vector<int> range(int upper)
{
    std::vector<int> ret;
    for (int ind0=0; ind0<upper; ind0++){
        ret.push_back(ind0);
    }
    return ret;
}

std::vector<int> range(int lower, int upper)
{
    std::vector<int> ret;
    for (int ind0=lower; ind0<upper; ind0++){
        ret.push_back(ind0);
    }
    return ret;
}

std::vector<int> range(int lower, int upper, int step)
{
    assert(step!=0);
    std::vector<int> ret;
    if (step>0){
        for (int ind0=lower; ind0<upper; ind0+=step){
            ret.push_back(ind0);
        }
    }else
    {
        for (int ind0=lower; ind0>upper; ind0+=step){
            ret.push_back(ind0);
        }
    }
    return ret;
}
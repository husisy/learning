extern "C" int hf1(int i)
{
    return i+1;
}

// g++ -shared -o tbd00/cppextension.so -fPIC cppextension.cpp

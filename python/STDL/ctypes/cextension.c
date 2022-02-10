#include <stdio.h>

// gcc -Wall -shared -Wl,-soname,cextension -o tbd00/cextension.so -fPIC cextension.c

void hf0()
{
    printf("hello world\n");
}


int hf1(int i)
{
    return i+1;
}


void hf2(double* np0, int N)
{
    for (int i=0; i<N; i++){
        np0[i] += 1;
    }
}

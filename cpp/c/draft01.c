#include <stdio.h>
#include <stdlib.h>


void test_rand()
{
    printf("\n# test_rand\n");
    printf("RAND_MAX = %d\n", RAND_MAX);
    printf("rand() = %d\n", rand());
    printf("rand() = %d\n", rand());
    printf("sizeof(float) = %d\n", (int)sizeof(float));
    printf("sizeof(double) = %d\n", (int)sizeof(double));
}

// gcc -o tbd00.exe draft_c_language.c
int main(int argc, char const *argv[])
{
    printf("\n# draft_c_language.c\n");
    return 0;
}

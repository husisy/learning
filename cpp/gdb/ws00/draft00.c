#include<stdio.h>
int main(int argc,char *argv[])
{
    if(1 >= argc)
    {
        printf("hello world!\n");
        return 0;
    }
    printf("hello World %s!\n",argv[1]);
    return 0 ;
}

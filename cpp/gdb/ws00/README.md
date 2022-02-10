# README

```bash
gcc -g -o tbd00.exe draft00.c
./tbd00.exe
./tbd00.exe 233

readelf -S tbd00.exe | grep debug

gdb tbd00.exe
# run
# set args 233
# break 9
# print argc
# print argv[0]
# print argv[1]
# display argc
# info display
# info registers
# list
# next
# step
# finish
# continue
# until 10
```

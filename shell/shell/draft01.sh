## basic
i=0 #no space
echo $i #no more echo below
((i++)) #i=1
let i++ #i=2
expr $i + 1 #3, space here
expr $i \* 2 #4, space here
echo $i 1 | awk '{printf $1+$2}' #3

##
expr 5 % 2 #1
let i=5%2 #1
echo 5 % 2 | bc
((i=5%2))

let i=5**2
((i=5**2))
echo "5^2" | bc

echo "obase=10;ibase=2;11" | bc -l

factor 100

# Internal Field Separator (IFS)
echo -n "$IFS" | od -c
echo -n "$IFS" | od -b
man ascii
# \t 011
# \n 012
#    040

echo "scale=3;1/13" | bc
echo "1 13" | awk '{printf("%0.3f\n",$1/$2)}'
echo 1/13100 | bc -l

##
type type #type is a shell builtin
type let #let is a shell builtin
type expr #expr is /usr/bin/expr
type bc #bs is /usr/bin/bc
type awk #awk is /usr/bin/awk
help type #help let
man expr #man bc, man awk

## while in one line
i=0
while [ $i -lt 100 ]; do ((i++)); done;
# while [ $i -lt 100 ]; do let i++; done;
# while [ $i -lt 100 ]; do i=$(expr $i + 1); done;
# while [ $i -lt 100 ]; do i=$(echo $i+1|bc); done;
# while [ $i -lt 100 ]; do i=$(echo "$i 1" | awk '{printf $1+$2}'); done;
echo $i

## while in script
i=0;
while [ $i -lt 100 ] #man test
do
    ((i++))
done
echo $i

##
echo $RANDOM
echo $RANDOM / 233
echo "" | awk '{srand(); printf("%f\n", rand())}'
echo "" | awk '{srand(); printf("%f\n", rand()*233)}'

##
seq 5
seq 1 5
seq 1 1 5
seq -s : 1 1 5
seq -w 9 1 11
seq -f "0x%g" 1 5
for i in `seq 1 5`; do echo $i; done
# for i in `seq -w 1 21`; do wget -c "http://thns.tsinghua.edu.cn/thnsebooks/ebook73/$1"; done
for i in {1..5}; do echo $i; done

##
wget -c http://tinylab.org
cat index.html | sed -e "s/[^a-zA-Z]/\n/g" | grep -v ^$ | sort | uniq -c | sort -n -k 1 -r | head -10

## generate random data
for i in $(seq 1 10)
do
    echo $i $(($RANDOM/8192+3)) $((RANDOM/10+3000))
done

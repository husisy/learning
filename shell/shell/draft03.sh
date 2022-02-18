hf0 () {
    echo "calling function hf0($1)"
}

hf0 () { for x in "$@"; do echo $x; done }
hf1 () { for x in $@; do echo $x; done }
z0="a b c"
for x in "$z0"; do echo $x; done #a b c
for x in $z0; do echo $x; done #a\nb\nc
hf0 a b c #a\nb\nc
hf1 a b c #a\nb\nc


# parameter expansion: https://stackoverflow.com/a/32343069
hf_colon () {
    : ${a:="helloA"} #optional variable
    : ${b:?variable b not set} #positional variable
    echo "a=${a}, b=${b}"
}
unset b
# hf_colon #raise error
b="helloB"
unset a
hf_colon #a=helloA, b=helloB
a="Ahello"
hf_colon #a=Ahello, b=helloB


## array
x=(2 23 233)
echo $x #only the first element
echo ${x[0]} #0-indexed
echo ${x[1]}
echo ${x[2]}
echo ${x[*]}
echo ${x[@]}
x[3]=2333 #append element
unset x[3]
unset x #unset all

## bash
echo {a..z} | tr " " "\n" > tbd00.txt
mapfile x < tbd00.txt
echo ${x[*]}
mapfile x < <(echo {a..z} | tr " " "\n")

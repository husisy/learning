printf "\n# test environment variable\n"
echo "BASH_VERSION is ${BASH_VERSION}"

printf "\n# test variable\n" #echo cannot print newline
tmp0="233"
echo "tmp0 is ${tmp0}"
echo ${tmp0}

for x in "2 23 233"; do echo $x; done #2 23 233
for x in 2 23 233; do echo $x; done #2\n23\n233
printf "\n# test for-loop\n"
for ((x=0; x<5; x++)) do
    echo "variable x=${x}"
done

for x in 2 23 233
do
    echo "variable x=${x}"
done

for x in `ls .`
do
    echo "file-i is ${x}"
done

for x in {1..5}
do
    echo "file-i is ${x}"
done

for x in {1..5..2}
do
    echo "file-i is ${x}"
done

for x in {1..5}
do
    echo "x is ${x}"
    if [ "${x}" == "3" ] #space before and after ==
    then
        echo "x==${x} is catched"
        break #continue
    fi
done

printf "\n# test string\n"
x0="~x0~"
echo "233"${x0}"233"
echo '233'"'"'233'
echo "string length of '${x0}' is ${#x0}"
echo "substring() is ${x0:1:2}"

printf "\n# done\n"


echo 5 | grep -q "^[0-9]$" && echo "true"
echo 2333 | grep "^[0-9]\+$" && echo "true"
echo "absSDGoiSDGdAnf" | grep "^[a-zA-Z]\+$" && echo "true"

echo -e "absSDGoiSDG2ddfAnf\nsadfasdFGa" | grep "^[a-zA-Z]\+$" && echo "true"
echo -e "absSDGoiSDG2ddfAnf\nsadfasdFGa" | grep "[a-zA-Z]\+" && echo "true"
echo -e "\t " | grep "[[:space:]]\+" && echo "true"
echo -e "\n" | grep "[[:print:]]" && echo "true"
echo -e "\t" | grep "[[:print:]]" && echo "true"

x1=233
echo ${#x1}
expr length $x1
echo $x1 | awk '{printf("%d\n", length($0))}'
echo -n $x1 | wc -c

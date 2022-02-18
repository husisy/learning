echo "run program $0 with $# arguments at pid $$"

true
echo $? #0
false
echo $? #1

false || echo "print" #print
true || echo "not-print"
true && echo "print" #print
false && echo "not-print"
false ; echo "print" #print

##
if true; then echo "print"; else echo "not-print"; fi
if false; then echo "not-print"; else echo "print"; fi
if [ true ]; then echo "print"; else echo "not-print"; fi
if true && false; then echo "not-print"; else echo "print"; fi
if true && true; then echo "print"; else echo "not-print"; fi
if true || false; then echo "print"; else echo "not-print"; fi
if false || false; then echo "not-print"; else echo "print"; fi
if [ true ] || [ false ]; then echo "print"; else echo "not-print"; fi
if ! false; then echo "print"; else echo "not-print"; fi
if ! true; then echo "not-print"; else echo "print"; fi

## if-then-else-fi in script
if true
then
    echo "print"
else
    echo "not-print"
fi

##
if test 5 -eq 5; then echo "print"; else echo "not-print"; fi
if test 5 -ne 5; then echo "not-print"; else echo "print"; fi
if [ 5 -eq 5 ]; then echo "print"; else echo "not-print"; fi
if [ 5 = 6 ]; then echo "not-print"; else echo "print"; fi

if test -n ""; then echo "YES"; else echo "NO"; fi
if test -n "233"; then echo "YES"; else echo "NO"; fi
if test -z ""; then echo "YES"; else echo "NO"; fi
if test -z "233"; then echo "YES"; else echo "NO"; fi
if ! test -n ""; then echo "YES"; else echo "NO"; fi
if [ "233" = "233" ]; then echo "YES"; else echo "NO"; fi


if test 5 -eq 5 -a 6 -eq 6; then echo "YES"; else echo "NO"; fi #and
if test 5 -eq 5 -o 5 -eq 6; then echo "YES"; else echo "NO"; fi #or
if test ! 5 -eq 6; then echo "YES"; else echo "NO"; fi #not

##
x1="233"
if [ $x1 = "233" ]; then echo "YES"; else echo "NO"; fi
if [ "$x1" = "233" ]; then echo "YES"; else echo "NO"; fi

##
([ true ] && [ true ]) && echo "true"


find . -name src -type d
find . -path '*/test/*.py' -type f
find . -mtime -1
find . -size +500k -size -10M -name '*.tar.gz'

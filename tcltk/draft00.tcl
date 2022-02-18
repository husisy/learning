puts "hello world"
puts {hello world}

# comment
puts "hello world"; #comment

# variable
set x0 hello
set x0 "hello wrold"
puts $x0
set x1 2.33
set x2 "$x0 $x1"
puts \$x0
puts \n_\t_
puts "hello \
world"
puts $
puts $$
puts $$x0
puts {$x0 $x1} ;#no substitute
puts "{$x0} $x1"
info exists x0

set x1 [set x0 "hello"]
set x1 "[set x0 world]"

# expression
expr {2+2}
set x0 233
expr {$x0 + 2}
set x0 2.1
set x0 3.
set x0 2.33E3
set x0 2.33e3
set x0 .233
set pi 3.1415926535
expr {sin($pi)}
expr {cos($pi)}
expr {1/2} ;#0
expr {1./2} ;#0.5
expr {double(1)/2} ;#0.5

set x 0
incr x
incr x 3

# string
set x0 "hello world"
string length $x0

# array
unset x0
set x0(1) 2
set x0(2) 3
set x0(3) 3
expr {($x0(1) + $x0(2)) * $x0(3)}

#if-else
if {1==2} {
    puts "hello"
} else {
    puts "world"
}
puts yes
puts no
expr {!yes}
expr {!no}
expr {true}
expr {!true}
expr {false}
expr {!false}

## for-loop
set ind0 0
while {$ind0 < 5} {
    puts $ind0
    set ind0 [expr {$ind0 + 1}]
}

for {set i 0} {$i < 5} {incr i} { puts $i }


## function
proc hf0 {} { puts "hello world" }
proc hf0 {arg0 arg1} {
    set ret [expr {$arg0 + $arg1}]
    return $ret
}
hf0 2 3

proc hf1 {arg0 {arg1 1}} {
    return [expr {$arg0 + $arg1}]
}
hf1 2 3
hf1 2

proc hfSetPositive {variable value} {
    upvar $variable myvar
    if {$value < 0} {
        set myvar [expr {-$value}]
    } else {
        set myvar $value
    }
    return $myvar
}
hfSetPositive x 5
puts $x
hfSetPositive x -5
puts $x

## list
set x0 "2 23 233"
lindex $x0 0 ;#0-index
lindex $x0 1
lindex $x0 2
lindex $x0 3 ;#empty output
llength $x0 ;#3
llength x0 ;#1

set x0 {{2} {23} {233}}
lindex $x0 0

set x0 [split "2/23/233" "/"]
set x0 [list 2 23 233]
foreach x1 $x0 { puts $x1 }
foreach {x1 x2} $x0 { puts "$x1,$x2" }

llength [list a b {c d e} {f {g h}}] ;#4
llength [lindex [list a b {c d e} {f {g h}}] 2] ;#3
llength [split "a b {c d e} {f {g h}}"] ;#8
concat a b {c d e} {f {g h}}
lappend {a b c} {d e}
# lappend
# linserrt
# lreplace
# lsearch
# lsort
# lrange

set x0 [list {Washington 1789} {Adams 1797} {Jefferson 1801} {Madison 1809} {Monroe 1817} {Adams 1825} ]
lsearch $x0 Washington*
lsearch $x0 Madison*
lrange $x0 0 3 ;#[left right]


## globbing pattern matching
string match f* foo
string match f?? foo
glob /tmp/*

## subcommand
set x0 "hello world"
llength $x0 ;#2
string length $x0 ;#11
string index $x0 0 ;#0-indexed
string range $x0 0 3 ;#[left right]
# string first
# string last
# string wordend
# string wordstart
# string tolower
# string toupper
# string trim

## regular expression
# regexp
# regsub

## associative arrays
unset x0
set x0(a) -1
set x0(b) -2
puts $x0(a)
puts $x0(b)
array exists x0
array names x0
array size x0
array get x0
parray x0
array unset x0
array set x0 [list a -1 b -2]


## dict
dict set clients 1 forenames Joe
dict set clients 1 surname   Schmoe
dict set clients 2 forenames Anne
dict set clients 2 surname   Other
puts "Number of clients: [dict size $clients]"
dict for {id info} $clients {
    puts "Client $id:"
    dict with info {
       puts "   Name: $forenames $surname"
    }
}
dict for {id info} $clients {
    dict for {id1 info1} $info {
        puts "$id $id1 $info1"
    }
}
dict keys clients

## file access
set fidout [open "tbd00.txt" w]
puts $fidout "hello"
puts $fidout "world"
close $fidout

set fidin [open "tbd00.txt" r]
while { [gets $fidin line] >= 0 } {
    puts $line
}
close $fidin

file normalize .

pwd

## run other program
# exec
# open
# subst


## info
info nameofexecutable
info commands
info globals
info locals
info vars
info procs
info exists x0
info functions
info tclversion
info cmdcount
info patchlevel
info script
info level
# info complete
pid


## misc
# source tbd00.tcl
# eval

## env
parray env

## special variable
puts $tcl_version
puts $tcl_pkgPath
puts $auto_path


## namespace, ensembles
set x0 233
puts $x0
puts $::x0

## errorInfo errorCode

## debugging trace

## command line arguments
puts $argc
puts $argv

## socket

## my
proc assert condition { if {![uplevel 1 expr $condition]} { return -code error "assertion failed: $condition" } }

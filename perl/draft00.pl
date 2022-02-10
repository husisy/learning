#! /usr/bin/perl
# perldoc perlintro
# perldoc -f functionname
use strict;
use warnings;

{
    print("\nspecial arguments \n");
    print('$_ @ARGV @_ %ENV', "\n");
    print("=================fengexian=================\n");
}

{
    print("\ndatatype - int/float/string\n");
    my $x1 = 233;
    my $x2 = 2.33;
    my $x3 = "233";
    print("int: $x1\n");
    print("float: $x2\n");
    print("int*float: ", $x1*$x2, "\n");
    print("string: $x3\n");
    print('string: $x3\n', "\n");
    print("string concat: ", $x3.$x3, "\n");
    print("=================fengexian=================\n");
}

{
    print("\ndataype - array \n");
    my @x1 = (2, 3, 3.3);
    print("array length: $#x1\n");
    print("@x1\n");
    print("$x1[0], $x1[1], $x1[2]\n");
    print("@x1[0..$#x1]\n");
    print("@x1[0,1,2]\n");
    print("@x1[0..2]\n");
    foreach(@x1){print("$_ ");}
    print("\n");

    my @x2 = reverse(@x1);
    print("reverse(): @x2\n");
    my @x3 = sort(@x2);
    print("sort(reverse()): @x3\n");
    print("=================fengexian=================\n");
}

{
    print("\n datatype - hash \n");
    my %x1 = (1=>"-1", 2=>"-2", 3=>"-3");
    my @x2 = keys(%x1);
    my @x3 = values(%x1);
    print("keys: @x2\n");
    print("values: @x3\n");
    print("=================fengexian=================\n");
}

{
    print("\n datetype-reference \n");
    my $x1 = {
        scalar => {description=>"single item", sigil=>'$'},
        array => {description=>"ordered list of items", sigil=>'@'},
        hash => {description=>"key/value pairs", sigil=>'%'},
    };
    print("scalar begin with a $x1->{'scalar'}->{'sigil'}\n");
    print("array  begin with a $x1->{'array'}->{'sigil'}\n");
    print("hash   begin with a $x1->{'hash'}->{'sigil'}\n");
    print("=================fengexian=================\n");
}

{
    print("\n conditional if-elsif-else \n");
    my $x1 = 233;
    if ($x1>=233) {print('$x1 >= 233: ', $x1>=233, "\n");}
    if ($x1<=233) {print('$x1 <= 233: ', $x1<=233, "\n");}
    print('$x1==233', "\n") if ($x1==233);
    print('unless $x1!=233', "\n") unless $x1!=233;
    print('unless not ($x1==233)', "\n") unless not $x1==233;
    print($x1==233, !($x1==233), "\n");
    print(!($x1==233), "233\n");
    print("=================fengexian=================\n");
}

{
    print("\nCollatz conjecture\n");
    my $x1 = int(rand(20)) + 10;
    while ($x1!=1){
        print("$x1\n");
        if ($x1%2==0){
            $x1 = int($x1/2);
        }else{
            $x1 = 3*$x1 + 1;
        }
    }
    print("=================fengexian=================\n");
}

{
    print("\nsubroutine \n");
    hf1("call hf1()\n");
    sub hf1 {print shift;}
    print("=================fengexian=================\n");
}

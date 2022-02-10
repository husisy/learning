'use strict';

// comment
/* still
comment */
var x1 = 233;
console.log(x1);
if (2>1) {x1 = 322;}
typeof(233);

// number
233;
2.33;
2.33e3;
-233;
NaN;
Infinity;
0xe9; //233

(1 + 2) * 5 / 2; //7.5
2 / 0; //Infinity
0 / 0; // NaN
10 % 3; //1
10.5 % 3; //1.5

// string
'abc';
"abc";
'\'';
"\"";
"\n\t\n\r\\";
'\x41'; //ASCII 'A'
'\u0041'; //utf-8 'A'
`233
233
233`; //backquote

'hello, ' + 233;
var x1 = 233;
`hello, ${x1}`;
'233'.length;
'233'[0];

'abc'.toUpperCase().toLowerCase();
'abc'.indexOf('bc')
'abc'.substring(0, 3)

// bool
true;
false;
2>1;
2===2; //never use "==" although js supports
2<1;

true && false;
true || false;
!true;
!false;
if (true) { alert('if1') }
if (false) { alert('never run'); } else { alert('if2'); }
NaN === NaN; //false
isNaN(NaN);
Math.abs(1-0.999)<0.1;

Boolean(null);
Boolean(undefined);
Boolean(0);
Boolean('');
Boolean(false);

//special
null;
undefined;

//array
[233,2.33,NaN,Infinity,'233',true,false,null,undefined,[]]
new Array(2,3,3);
[2,3,3][1]; //indexing start with 0

var x1 = [2,3,3];
x1.length;
x1[6] = 233;
x1.length;

for (var y of x1) { console.log(y); }
x1.forEach(function (element, index, array) { console.log(element.toString() + '::' + index.toString());});


//dict
var x1 = {
    name: '233',
    age: 233
};
x1.name;
x1['name'];
x1.name = '322';
delete x1.age;
'name' in x1;
'toString' in x1;
x1.hasOwnProperty('name');
x1.hasOwnProperty('toString');

//for-loop, while-loop, do-while-loop
var x1 = 0;
var i;
for (i=1; i<=100; i++){
    x1 = x1 + i;
}

//Map
var x1 = new Map([['Michael',95],['Bob',75],['Tracy',85]]);
// var x1 = new Map();
x1.get('Bob');
x1.set('Adam', 67);
x1.has('Adam');
x1.delete('Adam');
for (var y of x1){
    console.log(y[0]);
    console.log(y[1]);
}

//Set
var x1 = new Set([2,3,3]);
x1.add('2');
x1.delete('2');
for (var y of x1){ console.log(y); }
x1.forEach( function(y){ console.log(y);} );

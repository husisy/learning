//function
function myFunc0(x){
    if (typeof x !== 'number') {
        throw 'Not a number';
    }
    return 2*x;
}
myFunc0(-3);

var myFunc1 = function(x){
    return 2*x;
};
myFunc1(3);

function myFunc2(){
    for (var i=0; i<arguments.length; i++){
        console.log(String(i) + ': '+ arguments[i]);
    }
}
myFunc2(2,3,3,4);

function myFunc3(x, ...rest){
    console.log('0: '+x);
    for (var i=0; i<rest.length; i++){
        console.log(String(i) + ': '+ rest[i]);
    }
}
myFunc3(2,3,3,4);

//scope
var z1 = 233;
window.z1 = 234;

var MYSCOPE = {};
MYSCOPE.x1 = 233;

function myFunc4(){
    for (var tmp1=0; tmp1<100; tmp1++){}
    console.log(tmp1);
}
function myFunc5(){
    for (let tmp1=0; tmp1<100; tmp1++){}
    // console.log(tmp1); //error
}

//const
const PI = 3.14;

//Destructuring
var [x1,x2,x3] = [2,3,3]; //x1=2; x2=3; x3=3;
// var x1,x2,x3=[2,3,3]; //x1=undefined; x2=undefined; x3=[2,3,3];
var [[x1]] = [[233]];
var [,,x1] = [2,3,3];
var {x1,x2} = {
    x1:233,
    x2:'233'
};
var {x1:{x2}} = {x1: {x2:233}}; //x1(not defined); x2=233;


//generator
function* myFunc6(num1) {
    var x1=0,x2=1,ind1=0;
    while (ind1<num1){
        yield x1;
        [x1,x2] = [x2,x1+x2];
        ind1 ++;
    }
    return;
}
myFunc6(5).next();
for (var x of myFunc6(5)){ console.log(x); }

//class + object
var Student = {
    name:'Robot',
    height:1.2,
    run: function() { console.log(this.name+' is running...'); }
};
function createStudent(name){
    var s = Object.create(Student);
    s.name = name;
    return s;
}
var x1 = createStudent('x1');
x1.run();

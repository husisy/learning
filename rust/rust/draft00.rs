//#![allow(dead_code)]

use std::mem;

fn demo_print() {
    println!("\n# demo_print");

    print!("xxx\tyyy\nzzz");

    println!("str-{}, number-{}", "233", 233);
    println!("{0}{1}{1}", 2, 3);
    println!("{x}{y}{y}", x = 2, y = 3);
    println!("{:b}", 233);

    let x0 = 3;
    println!("integer x0={x0}");
    let x0 = "233";
    println!("stirng x0={x0}");

    let x1 = 5 + /* 90 + */ 5; //comment
    println!("x1={x1}"); //x1=10
}

fn _if_else_hf0(x: i32) {
    // let y: i32;
    // if x > 0 {
    //     y = x + 233;
    // } else {
    //     y = x - 233;
    // }
    let y = if x > 0 { x + 233 } else { x - 233 };
    println!("hf0({x})={y}");
}

fn demo_if_else() {
    println!("\n# demo_if_else");
    _if_else_hf0(-1);
    _if_else_hf0(0);
    _if_else_hf0(1);
}

fn demo_logical_operation() {
    println!("\n# demo_logical_operation");
    assert!(true);
    assert!(true == true);
    assert!(!false);
    assert!((1 + 2) == 3);
    assert_eq!(1 + 2, 3);
    assert!(true && true);
    assert!(true || false);
}

fn demo_loop() {
    println!("\n# demo_loop");
    let mut x = 0;
    let y = loop {
        x += 1;
        if x > 10 {
            break x * 2;
        }
    };
    println!("loop: x={x}, y={y}");

    let mut x = 3;
    while x != 0 {
        x -= 1;
    }
    println!("while x={x}");

    let x_list = [2, 23, 233];
    let mut y: i32 = 0;
    for x in x_list {
        y += x;
    }
    println!("for-in: y={y}");

    print!("for-range: ");
    for x in (1..4).rev() {
        print!("{x} ");
    }
    print!("\n");
}

fn _function_hf0(x: i32) -> i32 {
    x + 233
}

fn demo_function() {
    println!("\n# demo_function");
    println!("hf0(233)={}", _function_hf0(233));
}

fn _array_hf0(slice: &[i32]) {
    println!("slice[0]={}", slice[0]);
    println!("slice.len()={}", slice.len());
}

fn demo_array() {
    println!("\n# demo_array");

    // array are stack allocated
    let x0: [i32; 3] = [2, 3, 3];
    println!("x0[0]={}", x0[0]); //"{x0[0]}" is invalid
    println!("x0={:?}", x0); //Debug impl only for arrays of size less or equal to 32
    println!("x0={x0:?}");
    println!("x0.len()={}", x0.len());
    println!("mem::size_of_val(&x0)={} bytes", mem::size_of_val(&x0));
    let x1: [f32; 3] = [0.233; 3];
    println!("x1={:?}", x1);

    // borrow array as slice
    let x0: [i32; 5] = [0, 1, 2, 3, 4];
    _array_hf0(&x0);
    _array_hf0(&x0[1..4]); //[1,2,3]
    for i in 0..(x0.len() + 1) {
        match x0.get(i) {
            Some(x) => println!("x0.get({})={}", i, x),
            None => println!("x0.get({})=None", i),
        }
    }
    // x0[5]; //panic

    // empty array
    let x0: [i32; 0] = [];
    assert_eq!(&x0, &[]);
    assert_eq!(&x0, &[][..]); //just the same
}

fn demo_tuple() {
    println!("\n# demo_tuple");
    let x0 = (
        1u8, 2u16, 3u32, 4u64, -1i8, -2i16, -3i32, -4i64, 0.1f32, 0.2f64, 'a', true,
    );
    println!("x0={:?}", x0); //error if more than 12 elements
    println!("x0.0={}", x0.0);
}

fn demo_string() {
    println!("\n# demo_string");
    let x0 = String::from("hello");
    println!("x0={}", x0);
    let mut x1 = x0; //value borrowed here after move, x0 cannot be accessed anymore
    x1.push_str(" world!");
    // println!("x0={x0}"); //panic
    println!("x1={x1}");

    let x2 = &x1[..5]; // x2 is of type &str (immutable reference)
    let x3 = &x1[5..]; //&x1[5..s.len()]
    println!("x2={x2}, x3={x3}");
}

fn demo_reference(){
    println!("\n# demo_reference");
    let mut x0 = String::from("hello");
    {
        let x1 = &mut x0; //mutable reference
        // let x1:&mut String = &mut x0; //equivalent
        x1.push_str(" world!");
        println!("x1={}", x1);
    }
    println!("x0={}", x0);
}


fn _misc00_hf0(x: String) {
    println!("[hf0] {x}");
}

fn _misc00_hf1(x: &str) -> &str {
    &x[..]
    // &x
}

fn demo_misc00(){
    println!("\n# demo_misc00");
    _misc00_hf0("233".to_string()); //move? immutable reference?

    // let x = "helll world";
    println!("[hf1] {}", _misc00_hf1("helll world")); //move? immutable reference?
}

// rustc draft00.rs
// ./draft00
fn main() {
    println!("Hello world!");

    demo_misc00();

    demo_print();

    demo_if_else();

    demo_logical_operation();

    demo_loop();

    demo_function();

    demo_array();

    demo_tuple();

    demo_string();

    demo_reference();
}

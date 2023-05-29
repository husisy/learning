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

fn _if_else_hf0(x: i32){
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
    assert!(true==true);
    assert!(!false);
    assert!((1+2)==3);
    assert_eq!(1+2, 3);
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


fn _function_hf0(x:i32)->i32{
    x+233
}

fn demo_function(){
    println!("\n# demo_function");
    println!("hf0(233)={}", _function_hf0(233));
}


// rustc draft00.rs
// ./draft00
fn main() {
    println!("Hello world!");

    demo_print();

    demo_if_else();

    demo_logical_operation();

    demo_loop();

    demo_function();
}

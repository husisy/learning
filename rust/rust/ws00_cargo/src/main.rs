fn hf0(x: String) {
    println!("[hf0] {x}");
}

// i32
// char
// String

fn demo_function(x: i32) -> i32 {
    x + 233
}

fn demo_if_else(x: i32) {
    // let y: i32;
    // if x > 5 {
    //     y = x + 233;
    // } else {
    //     y = -x - 233;
    // }
    let y = if x > 0 { x + 233 } else { -x - 233 };
    println!("[demo_if_else] {}", y);
}

fn demo_loop() {
    let mut x = 0;
    let y = loop {
        x += 2;
        if x > 10 {
            break x * 2;
        }
    };
    println!("[demo_loop] {}", y);

    let mut x = 3;
    while x != 0 {
        x -= 1;
    }
    println!("[demo_loop] while {}", x);

    let x = [2, 23, 233];
    let mut y: i32 = 0;
    for xi in x {
        y += xi;
    }
    println!("[demo_loop] for-in {}", y);

    for x in (1..4).rev() {
        println!("[demo_loop] for-range {x}");
        // 3 2 1
    }
}

fn hf1(x: &str) -> &str {
    &x[..]
}

fn demo_logical_operation() {
    assert!(true);
    assert!(true==true);
    assert!(!false);
    assert!((1+2)==3);
    assert_eq!(1+2, 3);
    assert!(true && true);
    assert!(true || false);
}

fn main() {
    println!("Hello, world!");

    hf0("233".to_string());

    println!("[demo_function] {}", demo_function(233));

    demo_if_else(233);

    demo_loop();

    // let x = "helll world";
    println!("{}", hf1("helll world")); //TODO why pass???

    demo_logical_operation();
}

fn main() {
    println!("Hello world!");

    let tmp0 = 3;
    println!("integer tmp0={tmp0}");
    let tmp0 = "233";
    println!("stirng tmp0={tmp0}");

    println!("# std::format");
    println!("println! str-{}, number-{}", "233", 233);
    print!("print! \n");
    println!("{0}{1}{1}", 2, 3);
    println!("{x}{y}{y}", x=2, y=3);
    println!("{:b}", 233);

    let x0 = 5;
    println!("x0={}", x0); //x0=5

    let x1 = 5 + /* 90 + */ 5;
    println!("x1={}", x1); //x1=10
}

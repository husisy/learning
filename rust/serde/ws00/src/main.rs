use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct Point {
    x: i32,
    y: i32,
}

fn main() {
    let x0 = Point { x: 1, y: 2 };
    let x0_json_str = serde_json::to_string(&x0).unwrap(); //json string
    println!("x0_json_str: {}", x0_json_str); //{"x":1,"y":2}

    let x1: Point = serde_json::from_str(&x0_json_str).unwrap();
    println!("x1: {:?}", x1); //Point { x: 1, y: 2 }
}

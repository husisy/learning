mod draft_error_handling;

fn hf0(x: String) {
    println!("[hf0] {x}");
}

// i32
// char
// String

fn hf1(x: &str) -> &str {
    &x[..]
}


fn main() {
    hf0("233".to_string());

    // let x = "helll world";
    println!("{}", hf1("helll world")); //TODO why pass???

    draft_error_handling::demo_all();
}

mod draft_error_handling;
mod demo_rand;


fn main() {
    draft_error_handling::demo_all();

    // crate::demo_rand::demo_all(); //equivalent
    // use demo_rand::demo_all; demo_all(); //equivalent
    demo_rand::demo_all();
}

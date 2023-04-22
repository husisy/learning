use rand::Rng;
use std::io::Write; //bring flush() into scope

fn main() {
    println!("Hello, world!"); //this is comment

    let secret_number = rand::thread_rng().gen_range(1..=100); //[1,100] int32
    println!("secret number: {secret_number}");

    loop {
        print!("please input your guess: ");
        std::io::stdout().flush().unwrap();

        let mut guess = String::new();

        std::io::stdin()
            .read_line(&mut guess)
            .expect("failed to read line");

        let guess: u32 = match guess.trim().parse() {
            Ok(x) => x,
            Err(_) => continue,
        };
        println!("You guessed: {guess}");

        match guess.cmp(&secret_number) {
            std::cmp::Ordering::Less => println!("Too small!"),
            std::cmp::Ordering::Greater => println!("Too big!"),
            std::cmp::Ordering::Equal => {
                println!("you win!");
                break;
            }
        }
    }
}

// use rand::prelude::*;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::{Rng, SeedableRng, random, thread_rng};

pub fn demo_rand() {
    println!("\n#[demo_rand]");
    let x: u8 = random(); //unsigned 8-bit integer
    println!("[random()->u8] {}", x);

    if random() {
        println!("[random()->bool] true");
    } else {
        println!("[random()->bool] false");
    }

    let x: f64 = random(); //float64 [0,1)
    println!("[random()->f64] {}", x);

    // thread-local generator
    let mut rng = thread_rng();

    if rng.gen() {
        println!("[rng.gen()->bool] true");
    } else {
        println!("[rng.gen()->bool] false");
    }

    let x: f64 = rng.gen(); //float64 [0,1)
    println!("[rng.gen()->f64] {}", x);
    let x: f64 = rng.gen_range(-10.0..10.0); //float64 [-10,10)
    println!("[rng.gen_range(-10.0..10.0)->f64] {}", x);

    println!("[rng.gen::<i32>()] {}", rng.gen::<i32>()); //int32

    let x: i32 = rng.gen_range(1..3); // [1,3)
    println!("[rng.gen_range(1..3)->i32] {}", x);
    let x: i32 = rng.gen_range(1..=3); // [1,3]
    println!("[rng.gen_range(1..=3)->i32] {}", x);

    let distri = rand::distributions::Uniform::new_inclusive(1, 10); //integer [1,10]
    let mut arr0 = [0i32; 13]; //int32 with length=13
    for x in &mut arr0 {
        *x = rng.sample(distri);
    }
    println!("[rng.sample(distri)->i32] {:?}", arr0);

    let char_iter = "➡⬈⬆⬉⬅⬋⬇⬊".chars();
    let x = char_iter.choose(&mut rng).unwrap();
    println!("[char_iter.choose(&mut rng)] {}", x);

    let mut arr0 = [1, 2, 3, 4, 5];
    arr0.shuffle(&mut rng);
    println!("[arr0.shuffle(&mut rng)] {:?}", arr0);

    let mut rng_fixed = rand_chacha::ChaCha8Rng::seed_from_u64(233);
    let x: f64 = rng_fixed.gen(); //float 64-bit [0,1)
    println!("[rng_fixed.gen()->f64] {}", x);
}

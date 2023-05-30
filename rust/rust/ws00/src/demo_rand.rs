// use rand::prelude::*;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::{Rng, SeedableRng, random, thread_rng};

fn demo_basic() {
    println!("\n## demo_basic");
    let x_uint8: u8 = random();
    let x1_uint8 = random::<u8>();
    let x_bool: bool = random();
    let x_float64: f64 = random(); //[0,1)
    println!("random()->uint8: {x_uint8}");
    println!("random()->uint8: {x1_uint8}");
    println!("random()->bool: {x_bool}");
    println!("random()->float64: {x_float64}");

    if random() {
        println!("random()->bool: true");
    } else {
        println!("random()->bool: false");
    }

    // thread-local generator
    let mut rng = thread_rng();
    let x_uint8: u8 = rng.gen();
    let x1_uint8 = rng.gen::<u8>();
    let x_bool: bool = rng.gen();
    let x_float64: f64 = rng.gen(); //[0,1)
    println!("rng.gen()->uint8: {x_uint8}");
    println!("rng.gen()->uint8: {x1_uint8}");
    println!("rng.gen()->bool: {x_bool}");
    println!("rng.gen()->float64: {x_float64}");

    if rng.gen() {
        println!("rng.gen()->bool: true");
    } else {
        println!("rng.gen()->bool: false");
    }

    let x_float64: f64 = rng.gen_range(-10.0..10.0); //float64 [-10,10)
    let x_int32: i32 = rng.gen_range(1..3); // [1,3)
    let x1_int32: i32 = rng.gen_range(1..=3); // [1,3]
    println!("rng.gen_range(-10.0..10.0)->f64: {x_float64}");
    println!("rng.gen_range(1..3)->i32 {x_int32}");
    println!("rng.gen_range(1..=3)->i32: {x1_int32}");

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

pub fn demo_all(){
    println!("\n# demo_rand");
    demo_basic();
}

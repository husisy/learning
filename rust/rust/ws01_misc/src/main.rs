mod demo_rand;
mod demo_pmac;
mod demo_hmac;
mod demo_aead;
mod demo_tfhe;

use demo_pmac::demo_pmac_basic;

fn main() {
    println!("Hello, world!");

    demo_rand::demo_rand();

    // crate::demo_pmac::demo_pmac_basic(); //equivalent
    // demo_pmac::demo_pmac_basic(); //equivalent
    demo_pmac_basic();

    demo_hmac::demo_hmac_basic();

    demo_aead::demo_aes_siv_basic();

    demo_aead::demo_aes_siv_in_place();

    demo_tfhe::demo_tfhe_basic();
}

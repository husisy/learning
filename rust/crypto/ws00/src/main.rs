mod demo_pmac;
mod demo_hmac;
mod demo_aead;
mod demo_tfhe;

fn main() {
    demo_pmac::demo_all();

    demo_hmac::demo_all();

    demo_aead::demo_all();

    demo_tfhe::demo_all();
}

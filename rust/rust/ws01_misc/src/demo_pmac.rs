use hex::ToHex;
use aes::Aes128;
use pmac::{Pmac,Mac};

// https://github.com/RustCrypto/MACs
pub fn demo_pmac_basic() {
    println!("\n#[demo_pmac_basic]");

    // Create `Mac` trait implementation, namely PMAC-AES128
    let mut mac = Pmac::<Aes128>::new_from_slice(b"very secret key.").unwrap();
    mac.update(b"input message");

    // result(Output): wrapper around array of bytes for providing constant time equality check
    let result = mac.finalize();
    let tag_bytes = result.into_bytes();
    // println!("[tag_bytes] {:?}", tag_bytes);
    println!("[tag_bytes] {}", tag_bytes.encode_hex::<String>()); //don't use this to directly compare tags (timing attack)

    let mut mac = Pmac::<Aes128>::new_from_slice(b"very secret key.").unwrap();
    mac.update(b"input message");
    // `verify` will return `Ok(())` if tag is correct, `Err(MacError)` otherwise
    mac.verify(&tag_bytes).unwrap();
}

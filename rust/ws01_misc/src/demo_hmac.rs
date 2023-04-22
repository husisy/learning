use hex::ToHex;
use hex_literal::hex;
use hmac::{Hmac, Mac};
use sha2::Sha256;

// https://github.com/RustCrypto/MACs/tree/master/hmac
pub fn demo_hmac_basic() {
    println!("\n#[demo_hmac_basic]");

    // Create alias for HMAC-SHA256
    type HmacSha256 = Hmac<Sha256>;

    let mut mac = HmacSha256::new_from_slice(b"my secret and secure key")
        .expect("HMAC can take key of any size");
    mac.update(b"input message");

    // result(CtOutput): wrapper around array of bytes for providing constant time equality check
    let result = mac.finalize();
    let code_bytes = result.into_bytes();
    let expected = hex!("97d2a569059bbcd8ead4444ff99071f4c01d005bcefe0d3567e1be628e5fdcd9");
    assert_eq!(code_bytes[..], expected[..]);
    // println!("[code_bytes] {:?}", code_bytes);
    println!("[code_bytes] {}", code_bytes.encode_hex::<String>()); //don't use this to directly compare tags (timing attack)

    let mut mac = HmacSha256::new_from_slice(b"my secret and secure key")
        .expect("HMAC can take key of any size");
    mac.update(b"input message");
    let code_bytes = hex!("97d2a569059bbcd8ead4444ff99071f4c01d005bcefe0d3567e1be628e5fdcd9");
    // `verify_slice` will return `Ok(())` if code is correct, `Err(MacError)` otherwise
    mac.verify_slice(&code_bytes[..]).unwrap();
}

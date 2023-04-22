// https://github.com/RustCrypto/AEADs
use hex::ToHex;
use aes_siv::{Aes256SivAead, Nonce}; //Aes128SivAead
use aes_siv::aead::{Aead, KeyInit, OsRng, AeadInPlace};
// use heapless::Vec;
use aes_siv::aead::heapless::Vec;

pub fn demo_aes_siv_basic() {
    println!("\n#[demo_aes_siv_basic]");
    let key = Aes256SivAead::generate_key(&mut OsRng);
    let cipher = Aes256SivAead::new(&key);
    let nonce = Nonce::from_slice(b"any unique nonce"); // 128-bits; unique per message
    let ciphertext = cipher
        .encrypt(nonce, b"plaintext message".as_ref())
        .expect("encryption failure!");
    let plaintext = cipher
        .decrypt(nonce, ciphertext.as_ref())
        .expect("decryption failure!");
    assert_eq!(&plaintext, b"plaintext message");
    // println!("[ciphertext] {:?}", ciphertext);
    println!("[ciphertext] {}", ciphertext.encode_hex::<String>());
}


pub fn demo_aes_siv_in_place() {
    println!("\n#[demo_aes_siv_in_place]");
    let key = Aes256SivAead::generate_key(&mut OsRng);
    let cipher = Aes256SivAead::new(&key);
    let nonce = Nonce::from_slice(b"any unique nonce"); // 128-bits; unique per message

    let mut buffer: Vec<u8, 128> = Vec::new(); // Note: buffer needs 16-bytes overhead for auth tag tag
    buffer.extend_from_slice(b"plaintext message").expect("buffer too small");

    // Encrypt `buffer` in-place, replacing the plaintext contents with ciphertext
    cipher.encrypt_in_place(nonce, b"", &mut buffer).expect("encryption failure!");
    println!("[ciphertext] {}", buffer.encode_hex::<String>());
    // Decrypt `buffer` in-place, replacing its ciphertext context with the original plaintext
    cipher.decrypt_in_place(nonce, b"", &mut buffer).expect("decryption failure!");
    assert_eq!(&buffer, b"plaintext message");
}

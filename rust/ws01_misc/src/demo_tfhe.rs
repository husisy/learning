// use tfhe::boolean::prelude::*;
use tfhe::boolean::gen_keys;
use tfhe::boolean::server_key::{BinaryBooleanGates}; //ServerKey
// use tfhe::boolean::client_key::ClientKey;

pub fn demo_tfhe_basic() {
    println!("\n#[demo_tfhe_basic]");
    let (client_key, server_key) = gen_keys();

    // We use the client secret key to encrypt two messages:
    let ct_1 = client_key.encrypt(true);
    let ct_2 = client_key.encrypt(false);

    // We use the server public key to execute a boolean circuit:
    // if ((NOT ct_2) NAND (ct_1 AND ct_2)) then (NOT ct_2) else (ct_1 AND ct_2)
    let ct_3 = server_key.not(&ct_2);
    let ct_4 = server_key.and(&ct_1, &ct_2);
    let ct_5 = server_key.nand(&ct_3, &ct_4);
    let ct_6 = server_key.mux(&ct_5, &ct_3, &ct_4);

    // We use the client key to decrypt the output of the circuit:
    let output = client_key.decrypt(&ct_6);
    assert_eq!(output, true);
}


#[cfg(panic="unwind")]
fn _panic_config_hf0(){
    println!("[demo_panic_config] panic=unwind");
}

#[cfg(panic="abort")]
fn _panic_config_hf0(){
    println!("[demo_panic_config] panic=abort");
}

fn demo_panic_config(){
    _panic_config_hf0();
}

pub fn demo_all(){
    println!("\n# draft_error_handling.rs");
    demo_panic_config();
}

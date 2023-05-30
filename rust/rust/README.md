# rust

1. link
   * [official-site](https://www.rust-lang.org/)
   * [rust-doc](https://doc.rust-lang.org/std/)
   * [book/the-rust-programming-language](https://doc.rust-lang.org/book/)
   * [doc/rust-by-example](https://doc.rust-lang.org/stable/rust-by-example/)
   * [cargo-creates](https://crates.io/)
   * [github/sunface/rust-course](https://github.com/sunface/rust-course) rust中文教程
   * [rust-playground](https://play.rust-lang.org/)
   * [github/pyo3](https://github.com/PyO3)
2. install
   * mac/linux: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
   * uninstall `rustup self uninstall`
   * update `rustup update`
3. concept
   * word size, usize: `64bit` on x86-64,
   * string, string literal
   * ownership, reference, mutable reference, borrowing, slice, dereference
4. rule
   * each value in rust has an owner
   * only one owner at a time
   * when the owner goes out of scope, the value will be dropped
   * at any given time, either one mutable reference, or any number of immutable reference
   * reference must always be valid
5. short advice
   * use package: add `rand = "0.8"` to `[dependencies]` in `Cargo.toml`, add `use rand;` in `main.rs`
6. `cargo`
   * manifest file: `Cargo.toml`
   * package registry: [crates.io](https://crates.io/)
7. rust macro vs function
   * `println!`
8. misc
   * `.toml`: Tom's Obvious Minimal Language
   * crates
   * prelude
   * string: growable, UTF8
   * enumeration
   * trait
   * match, arm
   * shadowing
   * statement, expression
9.  function
   * parameter: 定义函数时写在括号里的东东
   * argument: 调用函数时写在括号里的东东

```rust
//type
u32 //unsigned integer
usize
i32 //integer
f32 //float
BigInt
'a' //character
"a" //string
true //boolean
() //unit type
```

```rust
#![allow(dead_code)]
```

package

1. link
   * [documentation](https://doc.rust-lang.org/book/ch07-00-managing-growing-projects-with-packages-crates-and-modules.html)
2. module system: package, crate, module, path, crate workspace
   * binary crate: `main`
   * library crate
   * package: `Cargo.toml`, `src/main.rs`, `src/lib.rs`, multiple binary crates `src/bin/`
   * module: `src/xxx.rs`, `src/xxx/mod.rs`
3. `self`, `super`, `crate`, `pub use`, `use xxx as yyy`

error handling

1. recovable errors: `Result<T,E>`
2. stop execution on errors: `panic!`
   * `panic!("crash and exit")`
   * `RUST_BACKTRACE=1`
3. unwinding, abort
4. `panic=abort/unwind`
   * `rustc -C panic=abort xxx.rs`
   * `Cargo.toml`: default to `unwind` in `[profile.dev]` and `[profile.release]`
5. `unimplemented`, `expect`, `unwrap`
6. `unwrap` returns a `panic` when it receives a `None`
   * `assert x is not None` (in Python)

## minimum working example

### mwe00

`draft00.rs`

```rust
fn main() {
   println!("Hello, world!");
}
```

1. compile `rustc draft00.rs`
2. run `./draft00`

### mwe01

```bash
cargo new ws00 #default enable git, unless already within a git folder
# cargo new ws00 --lib
cd ws00
cargo build
cargo run
# ./target/debug/ws00
cargo check
cargo build --release
# Cargo.lock

# cargo add rand --features small_rng
# cargo add rand rand_chacha
# cargo test
# cargo doc
# cargo publish
```

```toml
[package]
name = "ws00"
version = "0.1.0"
edition = "2021"

[dependencies]
```

## library

### rustfmt

### rand

1. link
   * [rust-rand/documentation](https://rust-random.github.io/book/intro.html)
   * [github/rust-rand](https://github.com/rust-random/rand)
   * [pcg-random.org/website](https://www.pcg-random.org/)
2. concept
   * True Random Number Generator (TRNG)
   * Pseudo Random Number Generator (PRNG)
   * Cryptographically Secure Pseudo Random Number Generator (CSPRNG)
   * Hardware Random Number Generator (HRNG)
   * entropy, `JitterRng`

### cryptography

1. link
   * awesome rust cryptography [link](https://cryptography.rs/#post-quantum-cryptography)
   * [github/awesome-rust-cryptography](https://github.com/rust-cc/awesome-cryptography-rust)
   * [github/pqcrypto](https://github.com/rustpq/pqcrypto)
   * [github/MACs](https://github.com/RustCrypto/MACs)

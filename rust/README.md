# rust

1. link
   * [official-site](https://www.rust-lang.org/)
   * [rust-doc](https://doc.rust-lang.org/std/)
   * [book/the-rust-programming-language](https://doc.rust-lang.org/book/)
   * [doc/rust-by-example](https://doc.rust-lang.org/stable/rust-by-example/)
   * [cargo-creates](https://crates.io/)
   * [github/sunface/rust-course](https://github.com/sunface/rust-course) rust中文教程
2. install
   * mac/linux: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
3. `rustup`
   * uninstall `rustup self uninstall`
   * update `rustup update`
4. `cargo`
   * `cargo build`
   * `cargo run`
   * `cargo test`
   * `cargo doc`
   * `cargo publish`
   * `cargo new hello-world` (default enable git, unless already within a git folder)
   * manifest file: `Cargo.toml`
   * package registry: [crates.io](https://crates.io/)
5. `rustfmt`
6. rust macro vs function
   * `println!`
7. misc
   * `.toml`: Tom's Obvious Minimal Language
   * crates
   * prelude
   * string: growable, UTF8
   * enumeration
   * trait
   * match, arm
   * shadowing
   * statement, expression
8. function
   * parameter: 定义函数时写在括号里的东东
   * argument: 调用函数时写在括号里的东东
9. type: `u32`, `f64`, `BigInt`

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
cargo new ws00_cargo #default enable git
# cargo new ws00_cargo --lib
cd ws00_cargo
cargo build
cargo run
# ./target/debug/ws00_cargo
cargo check
cargo build --release
# Cargo.lock
```

```toml
[package]
name = "ws00_cargo"
version = "0.1.0"
edition = "2021"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
```

## library

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

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

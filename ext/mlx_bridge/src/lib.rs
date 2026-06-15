//! C-ABI bridge between Ruby FFI and the mlx-rs Rust crate.
//!
//! Two kinds of exports leave this cdylib:
//!
//!   1. `mlx_rb_bridge_*` — bridge-owned helpers (ABI version probe,
//!      smoke tests, anything Rust-side we want to expose directly).
//!   2. Everything `mlx_*` from mlx-c — passed through. The linker
//!      pulls them from `mlx-sys`'s statically-linked libmlxc.a; the
//!      `exports.txt` whitelist + auto-generated `force_keep` slice in
//!      `build.rs` keep them from being dead-stripped.
//!
//! From Ruby's perspective the symbol table is byte-compatible with
//! the old libmlxc.dylib — `lib/mlx/ffi.rb` keeps its existing
//! `attach_function :mlx_*` calls.

use mlx_rs::Array;
use std::os::raw::c_int;

// Pulls in the auto-generated `force_keep` function, which takes the
// address of every mlx-c symbol the Ruby gem needs. The reference is
// what stops the linker's dead-strip from dropping libmlxc.a objects
// that nothing in Rust calls directly.
include!(concat!(env!("OUT_DIR"), "/force_keep.rs"));

/// Bridge ABI version. `lib/mlx/ffi.rb` reads this at load time and
/// refuses to attach if it doesn't match the gem's expectations.
pub const BRIDGE_ABI: u32 = 1;

#[no_mangle]
pub extern "C" fn mlx_rb_bridge_abi_version() -> u32 {
    // Force a load-time reference to the force-keep symbol so the
    // linker considers it live. The return value is ignored.
    let _ = mlx_rb_bridge_force_keep();
    BRIDGE_ABI
}

/// Smoke test: build two NxN ones matrices in mlx-rs and matmul them.
/// Returns the mean (= N for ones-matmul) rounded to i64; `i64::MIN`
/// on any error.
#[no_mangle]
pub extern "C" fn mlx_rb_bridge_smoke_matmul(n: c_int) -> i64 {
    if n <= 0 {
        return i64::MIN;
    }
    let n = n as i32;
    let a = match Array::ones::<f32>(&[n, n]) {
        Ok(arr) => arr,
        Err(_) => return i64::MIN,
    };
    let b = match Array::ones::<f32>(&[n, n]) {
        Ok(arr) => arr,
        Err(_) => return i64::MIN,
    };
    let c = match a.matmul(&b) {
        Ok(arr) => arr,
        Err(_) => return i64::MIN,
    };
    let scalar = match c.mean(None) {
        Ok(arr) => arr,
        Err(_) => return i64::MIN,
    };
    scalar.item::<f32>().round() as i64
}

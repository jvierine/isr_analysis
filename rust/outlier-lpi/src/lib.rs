//! Reusable Rust outlier-resistant lag-profile inversion.
//! Signed convention R(l) = x(t) conj(x(t+l)). No SVD or missing-lag fill.
pub type C = num_complex::Complex64;
pub mod signal;
pub mod lpi;
pub use lpi::*;

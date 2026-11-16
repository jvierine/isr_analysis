//! SYISR AC campaign analysis. Signed products, UTC metadata, nominal ray geometry.
//! Based on the verified 2026-11-16 gated pilot; error resampling is at code-cycle level.
pub use num_complex::Complex64 as Complex;
pub mod reader;
pub mod theory;
pub mod coupled;
pub mod exact_response;
pub mod geomagnetic;
pub mod campaign;
pub mod screen;
pub mod decode;
pub mod fitting;

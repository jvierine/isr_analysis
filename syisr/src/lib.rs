//! SYISR AC campaign analysis. Signed products, UTC metadata, nominal ray geometry.
//! Based on the verified 2026-11-16 gated pilot; error resampling is at code-cycle level.
pub use num_complex::Complex64 as Complex;
pub mod campaign;
pub mod coupled;
pub mod decode;
pub mod exact_response;
pub mod fitting;
pub mod geomagnetic;
pub mod reader;
pub mod screen;
pub mod theory;

pub mod grid;

pub mod spline;
pub mod ac_range;

pub mod fullprofile;

pub mod response_dot;

pub mod initialization;

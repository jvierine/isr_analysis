//! Ion-line physics and inference. All state is supplied explicitly; no I/O or globals.
pub mod dft;
pub mod beata;
pub mod fit;
pub mod noise;
pub mod multirate;
pub mod precipitation;
pub mod spectrum;
pub use num_complex::Complex64 as Complex;

pub mod clutter;

pub mod voltage;

pub mod full_profile;

pub mod correlation;

pub mod pulse_response;

pub mod composition;

pub mod topside;

pub mod geomagnetic;

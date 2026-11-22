//! SYISR AC campaign analysis. Signed products, UTC metadata, nominal ray geometry.
//! Based on the verified 2026-11-16 gated pilot; error resampling is at code-cycle level.
pub use num_complex::Complex64 as Complex;
pub mod campaign;
pub mod coupled;
pub mod decode;
pub mod exact_response;
pub mod receiver_filter;
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
pub mod h_transition;

pub mod sharedcomposition;

mod autodiff;

pub mod drift;

pub mod lp480;
#[path = "../ulp2ms/src/iqr_screen.rs"]
pub mod lp480_screen;
pub mod lp480_fit;
pub mod lp480_profile;
pub mod lp480_l2;
pub mod lp480_extend;

pub mod lower_hybrid;

pub mod lfm_geometry;

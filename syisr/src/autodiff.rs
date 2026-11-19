//! Six-direction forward-mode AD for the local physical spectrum.
//! The full profile Jacobian retains the existing analytic spline/composition chain rule.
use num_complex::Complex64 as C;
use std::ops::{Add, Div, Mul, Neg, Sub};
pub(crate) const N: usize = 6;
#[derive(Clone, Copy, Debug)]
pub(crate) struct D {
    pub v: f64,
    pub d: [f64; N],
}
impl D {
    pub fn constant(v: f64) -> Self {
        Self { v, d: [0.; N] }
    }
    pub fn seed(v: f64, k: usize, slope: f64) -> Self {
        let mut x = Self::constant(v);
        x.d[k] = slope;
        x
    }
    pub fn sqrt(self) -> Self {
        let v = self.v.sqrt();
        Self {
            v,
            d: self.d.map(|d| d / (2. * v)),
        }
    }
}
impl Add for D {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: std::array::from_fn(|i| self.d[i] + b.d[i]),
        }
    }
}
impl Sub for D {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            v: self.v - b.v,
            d: std::array::from_fn(|i| self.d[i] - b.d[i]),
        }
    }
}
impl Mul for D {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: std::array::from_fn(|i| self.d[i] * b.v + self.v * b.d[i]),
        }
    }
}
impl Div for D {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        let v = self.v / b.v;
        Self {
            v,
            d: std::array::from_fn(|i| (self.d[i] - v * b.d[i]) / b.v),
        }
    }
}
impl Neg for D {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            v: -self.v,
            d: self.d.map(|v| -v),
        }
    }
}
impl Mul<f64> for D {
    type Output = Self;
    fn mul(self, b: f64) -> Self {
        Self {
            v: self.v * b,
            d: self.d.map(|v| v * b),
        }
    }
}
impl Div<f64> for D {
    type Output = Self;
    fn div(self, b: f64) -> Self {
        self * (1. / b)
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) struct Z {
    pub v: C,
    pub d: [C; N],
}
impl Z {
    pub fn constant(v: C) -> Self {
        Self {
            v,
            d: [C::default(); N],
        }
    }
    pub fn i(x: D) -> Self {
        Self {
            v: C::new(0., x.v),
            d: x.d.map(|v| C::new(0., v)),
        }
    }
    pub fn re(self) -> D {
        D {
            v: self.v.re,
            d: self.d.map(|v| v.re),
        }
    }
    pub fn norm_sqr(self) -> D {
        D {
            v: self.v.norm_sqr(),
            d: self.d.map(|v| 2. * (self.v.conj() * v).re),
        }
    }
}
impl Add for Z {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            v: self.v + b.v,
            d: std::array::from_fn(|i| self.d[i] + b.d[i]),
        }
    }
}
impl Mul for Z {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        Self {
            v: self.v * b.v,
            d: std::array::from_fn(|i| self.d[i] * b.v + self.v * b.d[i]),
        }
    }
}
impl Mul<D> for Z {
    type Output = Self;
    fn mul(self, b: D) -> Self {
        Self {
            v: self.v * b.v,
            d: std::array::from_fn(|i| self.d[i] * b.v + self.v * b.d[i]),
        }
    }
}

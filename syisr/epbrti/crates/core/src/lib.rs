//! Physical mapping and editing core. u/v are IGRF reference-plane coordinates;
//! the fitted field is a relative echo proxy, never an absolute density.
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
pub struct Editor {
    nx: usize,
    ny: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    q: Vec<f32>,
    weights: Vec<f32>,
    seed: Vec<f32>,
    support: Vec<f32>,
}
#[wasm_bindgen]
impl Editor {
    #[wasm_bindgen(constructor)]
    pub fn new(
        nx: usize,
        ny: usize,
        x0: f64,
        y0: f64,
        dx: f64,
        seed: Vec<f32>,
        support: Vec<f32>,
    ) -> Result<Editor, JsValue> {
        if nx < 2 || ny < 2 || dx <= 0. || seed.len() != 3 * nx * ny || support.len() != seed.len()
        {
            return Err(JsValue::from_str("Invalid grid"));
        }
        Ok(Self {
            nx,
            ny,
            x0,
            y0,
            dx,
            weights: vec![0.; seed.len()],
            q: seed.clone(),
            seed,
            support,
        })
    }
    pub fn grid(&self, g: usize) -> Vec<f32> {
        self.q[g * self.nx * self.ny..(g + 1) * self.nx * self.ny].to_vec()
    }
    pub fn state(&self) -> Vec<f32> {
        [self.q.as_slice(), self.weights.as_slice()].concat()
    }
    pub fn restore(&mut self, v: Vec<f32>) -> bool {
        if v.len() != self.q.len() * 2 || v.iter().any(|v| !v.is_finite()) {
            return false;
        }
        let n = self.q.len();
        if v[..n].iter().any(|v| *v < 0. || *v > 1.) {
            return false;
        }
        self.q.copy_from_slice(&v[..n]);
        self.weights.copy_from_slice(&v[n..]);
        true
    }
    pub fn reset(&mut self, g: usize) {
        let n = self.nx * self.ny;
        self.q[g * n..(g + 1) * n].copy_from_slice(&self.seed[g * n..(g + 1) * n]);
        self.weights[g * n..(g + 1) * n].fill(0.);
    }
    pub fn paint(&mut self, g: usize, u: f64, v: f64, radius: f64, amount: f64) -> usize {
        if g >= 3 || radius <= 0. || !amount.is_finite() {
            return 0;
        }
        let mut count = 0;
        let ix = ((u - self.x0) / self.dx).round() as isize;
        let iy = ((v - self.y0) / self.dx).round() as isize;
        let n = (radius / self.dx).ceil() as isize;
        for j in (iy - n).max(0)..=(iy + n).min(self.ny as isize - 1) {
            for i in (ix - n).max(0)..=(ix + n).min(self.nx as isize - 1) {
                let d = ((self.x0 + i as f64 * self.dx - u).powi(2)
                    + (self.y0 + j as f64 * self.dx - v).powi(2))
                .sqrt()
                    / radius;
                if d > 1. {
                    continue;
                }
                let k = g * self.nx * self.ny + j as usize * self.nx + i as usize;
                if self.support[k] <= 0. {
                    continue;
                }
                let w = (1. - d * d).powi(2) * amount;
                self.weights[k] += w as f32;
                self.q[k] = (self.q[k] - w as f32).clamp(0., 1.);
                count += 1;
            }
        }
        count
    }
    pub fn sample(&self, g: usize, u: f64, v: f64) -> f64 {
        sample(
            &self.q,
            &self.support,
            g,
            self.nx,
            self.ny,
            self.x0,
            self.y0,
            self.dx,
            u,
            v,
        )
    }
    /// geometry layout per range: altitude, centre u/v, nine quadrature u/v pairs.
    pub fn predict_linear(
        &self,
        g: usize,
        nt: usize,
        t0: f64,
        dt: f64,
        ranges: Vec<f32>,
        geometry: Vec<f32>,
        reference: Vec<f32>,
        vx: f64,
        vy: f64,
        tref: f64,
    ) -> Vec<f32> {
        let nr = ranges.len();
        let mut out = vec![f32::NAN; nt * nr];
        if geometry.len() != nr * 21 || reference.len() != nr {
            return out;
        }
        let w = [1. / 6., 2. / 3., 1. / 6.];
        for j in 0..nr {
            if !reference[j].is_finite() || reference[j] <= 0. {
                continue;
            }
            for i in 0..nt {
                let t = t0 + i as f64 * dt - tref;
                let mut ratio = 0.;
                let mut valid = true;
                for k in 0..9 {
                    let a = self.sample(
                        g,
                        geometry[j * 21 + 3 + k * 2] as f64 - vx * t,
                        geometry[j * 21 + 4 + k * 2] as f64 - vy * t,
                    );
                    if !a.is_finite() {
                        valid = false;
                        break;
                    }
                    ratio += w[k / 3] * w[k % 3] * a;
                }
                if valid {
                    let p = ratio * reference[j] as f64 * (ranges[j] as f64).powi(2);
                    out[j * nt + i] = p as f32;
                }
            }
        }
        out
    }
    pub fn predict(
        &self,
        g: usize,
        nt: usize,
        t0: f64,
        dt: f64,
        ranges: Vec<f32>,
        geometry: Vec<f32>,
        reference: Vec<f32>,
        vx: f64,
        vy: f64,
        tref: f64,
    ) -> Vec<u8> {
        let nr = ranges.len();
        let mut out = vec![0; nt * nr];
        if geometry.len() != nr * 21 || reference.len() != nr {
            return out;
        }
        let w = [1. / 6., 2. / 3., 1. / 6.];
        for j in 0..nr {
            if !reference[j].is_finite() || reference[j] <= 0. {
                continue;
            }
            for i in 0..nt {
                let t = t0 + i as f64 * dt - tref;
                let mut ratio = 0.;
                let mut valid = true;
                for k in 0..9 {
                    let a = self.sample(
                        g,
                        geometry[j * 21 + 3 + k * 2] as f64 - vx * t,
                        geometry[j * 21 + 4 + k * 2] as f64 - vy * t,
                    );
                    if !a.is_finite() {
                        valid = false;
                        break;
                    }
                    ratio += w[k / 3] * w[k % 3] * a;
                }
                if valid {
                    let p = ratio * reference[j] as f64 * (ranges[j] as f64).powi(2);
                    let db = if p > 0. { 10. * p.log10() } else { 78. };
                    out[j * nt + i] =
                        (((db - 78.) / 27. * 254.).round().clamp(0., 254.) + 1.) as u8;
                }
            }
        }
        out
    }
}
fn sample(
    q: &[f32],
    support: &[f32],
    g: usize,
    nx: usize,
    ny: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    u: f64,
    v: f64,
) -> f64 {
    let x = (u - x0) / dx;
    let y = (v - y0) / dx;
    if g >= 3 || x < 0. || y < 0. || x >= nx as f64 - 1. || y >= ny as f64 - 1. {
        return f64::NAN;
    }
    let i = x.floor() as usize;
    let j = y.floor() as usize;
    let fx = x - i as f64;
    let fy = y - j as f64;
    let mut z = 0.;
    for (a, b, w) in [
        (0, 0, (1. - fx) * (1. - fy)),
        (1, 0, fx * (1. - fy)),
        (0, 1, (1. - fx) * fy),
        (1, 1, fx * fy),
    ] {
        let k = g * nx * ny + (j + b) * nx + i + a;
        if w > 1e-6 && support[k] <= 0. {
            return f64::NAN;
        }
        z += q[k] as f64 * w;
    }
    z
}
#[wasm_bindgen]
pub fn map_hover(
    r: f64,
    t: f64,
    ranges: Vec<f32>,
    geom: Vec<f32>,
    vx: f64,
    vy: f64,
    tref: f64,
) -> Vec<f64> {
    if ranges.len() < 2
        || geom.len() != ranges.len() * 21
        || r < ranges[0] as f64
        || r > *ranges.last().unwrap() as f64
    {
        return vec![];
    }
    let i = ranges
        .partition_point(|x| (*x as f64) <= r)
        .saturating_sub(1)
        .min(ranges.len() - 2);
    let f = (r - ranges[i] as f64) / (ranges[i + 1] - ranges[i]) as f64;
    let at = |k| geom[i * 21 + k] as f64 * (1. - f) + geom[(i + 1) * 21 + k] as f64 * f;
    vec![at(1) - vx * (t - tref), at(2) - vy * (t - tref), at(0)]
}
#[wasm_bindgen]
pub fn cross(
    u: f64,
    v: f64,
    ranges: Vec<f32>,
    geom: Vec<f32>,
    vx: f64,
    vy: f64,
    tref: f64,
    tmin: f64,
    tmax: f64,
) -> Vec<f64> {
    if geom.len() != ranges.len() * 21 {
        return vec![];
    }
    let mut roots = vec![];
    for i in 0..ranges.len() - 1 {
        // also valid if drift has an upward component
        let residual = |j: usize| {
            let dt = if vx.abs() > 1e-10 {
                (geom[j * 21 + 1] as f64 - u) / vx
            } else {
                0.
            };
            geom[j * 21 + 2] as f64 - v - vy * dt
        };
        let a = residual(i);
        let b = residual(i + 1);
        if a * b > 0. || (a - b).abs() < 1e-12 {
            continue;
        }
        let f = a / (a - b);
        let at = |k| geom[i * 21 + k] as f64 * (1. - f) + geom[(i + 1) * 21 + k] as f64 * f;
        let t = tref + (at(1) - u) / vx;
        if t >= tmin && t <= tmax {
            roots.extend([
                t,
                ranges[i] as f64 * (1. - f) + ranges[i + 1] as f64 * f,
                at(0),
            ]);
        }
    }
    roots
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn brush_and_restore() {
        let n = 9 * 9 * 3;
        let mut e = Editor::new(9, 9, -4., -4., 1., vec![0.7; n], vec![1.; n]).unwrap();
        e.paint(0, 0., 0., 2., 0.2);
        assert!((e.sample(0, 0., 0.) - 0.5).abs() < 1e-6);
        assert!((e.sample(1, 0., 0.) - 0.7).abs() < 1e-6);
        let s = e.state();
        e.reset(0);
        assert!(e.restore(s));
        assert!((e.sample(0, 0., 0.) - 0.5).abs() < 1e-6);
    }
    #[test]
    fn roundtrip_mapping_with_drift() {
        let r = vec![200., 400., 600.];
        let mut g = vec![0.; 63];
        for j in 0..3 {
            g[j * 21] = r[j];
            g[j * 21 + 1] = j as f32 * 20.;
            g[j * 21 + 2] = j as f32 * 100.;
        }
        let a = map_hover(350., 1050., r.clone(), g.clone(), 0.12, 0., 1000.);
        let b = cross(a[0], a[1], r, g, 0.12, 0., 1000., 0., 2000.);
        assert!((b[0] - 1050.).abs() < 1e-8);
        assert!((b[1] - 350.).abs() < 1e-8);
    }
    #[test]
    fn linear_prediction_keeps_physical_scale() {
        let e = Editor::new(9, 9, -4., -4., 1., vec![0.5; 243], vec![1.; 243]).unwrap();
        let g = vec![0.; 21];
        let p = e.predict_linear(0, 2, 0., 1., vec![2.], g.clone(), vec![100.], 0., 0., 0.);
        assert_eq!(p, vec![200., 200.]);
        let ne = e.predict_linear(0, 1, 0., 1., vec![1.], g.clone(), vec![1e12], 0., 0., 0.);
        assert!((ne[0] as f64 / 5e11 - 1.).abs() < 1e-6);
        let missing = e.predict_linear(0, 1, 0., 1., vec![1.], g, vec![f32::NAN], 0., 0., 0.);
        assert!(missing[0].is_nan());
    }
}

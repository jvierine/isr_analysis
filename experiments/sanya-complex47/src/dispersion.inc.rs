// Collisionless hot susceptibility: causal dispersion relation on SIGNED frequencies.
// Linear FFT convolution evaluates the nonperiodic, discrete principal value.
// No odd-frequency assumption: pitch-anisotropic distributions need both signs.
use rustfft::{num_complex::Complex, FftPlanner};
#[derive(Clone, Copy)]
struct DispersionGrid {
    du: f64,
    extent: f64,
    dense_edge: f64,
    middle_edge: f64,
    middle_stride: usize,
    outer_stride: usize,
}
impl DispersionGrid {
    fn standard() -> Self {
        Self {
            du: 0.00025,
            extent: 96.,
            dense_edge: 8.5,
            middle_edge: 20.,
            middle_stride: 4,
            outer_stride: 16,
        }
    }
    fn refined() -> Self {
        Self {
            du: 0.000125,
            extent: 112.,
            dense_edge: 8.5,
            middle_edge: 20.,
            middle_stride: 4,
            outer_stride: 16,
        }
    }
    fn n(&self) -> usize {
        2 * (self.extent / self.du).round() as usize + 1
    }
    fn coordinate(&self, i: usize) -> f64 {
        (i as f64 - (self.n() / 2) as f64) * self.du
    }
    fn sample_indices(&self) -> Vec<usize> {
        let n = self.n();
        let mid = n / 2;
        let mut v = vec![];
        for i in 0..n {
            let u = self.coordinate(i).abs();
            let stride = if u <= self.dense_edge {
                1
            } else if u <= self.middle_edge {
                self.middle_stride
            } else {
                self.outer_stride
            };
            if (i as isize - mid as isize).unsigned_abs() % stride == 0 || i == 0 || i == n - 1 {
                v.push(i);
            }
        }
        v
    }
}
struct CausalTransform {
    n: usize,
    fft_len: usize,
    kernel: Vec<Complex<f64>>,
    forward: std::sync::Arc<dyn rustfft::Fft<f64>>,
    inverse: std::sync::Arc<dyn rustfft::Fft<f64>>,
}
impl CausalTransform {
    fn new(n: usize) -> Self {
        let len = (2 * n).next_power_of_two();
        let mut planner = FftPlanner::new();
        let forward = planner.plan_fft_forward(len);
        let inverse = planner.plan_fft_inverse(len);
        let mut kernel = vec![Complex::new(0., 0.); len];
        for k in (1..n).step_by(2) {
            let h = -2. / (PI * k as f64);
            kernel[k].re = h;
            kernel[len - k].re = -h;
        }
        forward.process(&mut kernel);
        Self {
            n,
            fft_len: len,
            kernel,
            forward,
            inverse,
        }
    }
    fn real_from_imag(&self, im: &[f64]) -> Vec<f64> {
        assert_eq!(im.len(), self.n);
        let mut work = vec![Complex::new(0., 0.); self.fft_len];
        for (v, &x) in work.iter_mut().zip(im) {
            v.re = x;
        }
        self.forward.process(&mut work);
        for (x, k) in work.iter_mut().zip(&self.kernel) {
            *x *= k;
        }
        self.inverse.process(&mut work);
        work[..self.n]
            .iter()
            .map(|x| x.re / self.fft_len as f64)
            .collect()
    }
}
fn signed_hot(
    local: &LocalPe,
    c: &Core,
    grid: DispersionGrid,
    transform: &CausalTransform,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let indices = grid.sample_indices();
    let gyro = E * c.b / ME;
    let samples: Vec<_> = indices
        .par_iter()
        .map(|&i| {
            let (m, im) = local.terms(c, grid.coordinate(i) * gyro);
            (m, im)
        })
        .collect();
    let n = grid.n();
    let mut im = vec![0.; n];
    let mut source = vec![0.; n];
    for j in 0..indices.len() - 1 {
        let l = indices[j];
        let r = indices[j + 1];
        for i in l..=r {
            let t = (i - l) as f64 / (r - l) as f64;
            source[i] = samples[j].0 * (1. - t) + samples[j + 1].0 * t;
            im[i] = samples[j].1 * (1. - t) + samples[j + 1].1 * t;
        }
    }
    let re = transform.real_from_imag(&im);
    (source, im, re)
}
fn interpolate(grid: DispersionGrid, v: &[f64], u: f64) -> f64 {
    let q = u / grid.du + (grid.n() / 2) as f64;
    assert!(q >= 0. && q < (v.len() - 1) as f64);
    let j = q.floor() as usize;
    let t = q - j as f64;
    v[j] * (1. - t) + v[j + 1] * t
}

struct PeArchive {
    metadata: serde_json::Value,
    energy: Vec<f64>,
    mu: Vec<f64>,
    wg: Vec<f64>,
    phase: Vec<f64>,
    height: Vec<f64>,
    nh: Vec<f64>,
}
impl PeArchive {
    fn load(path: &str) -> Result<Self> {
        let h = hdf5::File::open(path)?;
        let meta: serde_json::Value =
            serde_json::from_slice(&h.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
        let height = meta["height_km"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect();
        Ok(Self {
            metadata: meta.clone(),
            energy: h.dataset("energy_ev")?.read_raw::<f64>()?,
            mu: h.dataset("mu")?.read_raw::<f64>()?,
            wg: h.dataset("mu_weight")?.read_raw::<f64>()?,
            phase: h
                .dataset("phase_density_s3_m6")?
                .read_raw::<f32>()?
                .iter()
                .map(|&x| x as f64)
                .collect(),
            height,
            nh: h.dataset("photoelectron_density_m3")?.read_raw::<f64>()?,
        })
    }
    fn local(&self, h: f64) -> LocalPe {
        let j = (self.height.partition_point(|&x| x < h))
            .saturating_sub(1)
            .min(self.height.len() - 2);
        let u = ((h - self.height[j]) / (self.height[j + 1] - self.height[j])).clamp(0., 1.);
        let stride = self.energy.len() * self.mu.len();
        {
            let mut energy = vec![0.001];
            energy.extend(&self.energy);
            energy.push(650.);
            let mut phase = vec![0.; self.mu.len()];
            phase.extend((0..stride).map(|i| {
                self.phase[j * stride + i] * (1. - u) + self.phase[(j + 1) * stride + i] * u
            }));
            phase.extend(vec![0.; self.mu.len()]);
            LocalPe {
                energy,
                mu: self.mu.clone(),
                phase,
                nh: self.nh[j] * (1. - u) + self.nh[j + 1] * u,
            }
        }
    }
}
struct LocalPe {
    energy: Vec<f64>,
    mu: Vec<f64>,
    phase: Vec<f64>,
    nh: f64,
}
impl LocalPe {
    // Positive phase-density interpolation: linear differential density in energy and mu.
    // Partial derivatives are those of precisely this reconstruction.
    fn evaluate(&self, e: f64, mu: f64) -> (f64, f64, f64) {
        if e < self.energy[0] || e > *self.energy.last().unwrap() {
            return (0., 0., 0.);
        }
        let ie = self
            .energy
            .partition_point(|&x| x < e)
            .saturating_sub(1)
            .min(self.energy.len() - 2);
        let im = self
            .mu
            .partition_point(|&x| x < mu)
            .saturating_sub(1)
            .min(self.mu.len() - 2);
        let ee0 = self.energy[ie];
        let ee1 = self.energy[ie + 1];
        let t = (e - ee0) / (ee1 - ee0);
        let m = mu.clamp(self.mu[0], *self.mu.last().unwrap());
        let u = (m - self.mu[im]) / (self.mu[im + 1] - self.mu[im]);
        let nm = self.mu.len();
        // Reconstruct differential number density, proportional to sqrt(E)*f_v,
        // linearly in energy; this preserves positivity and reduces group-population interpolation bias.
        let energy_eval = |k: usize| {
            let a = self.phase[ie * nm + k] * ee0.sqrt();
            let b = self.phase[(ie + 1) * nm + k] * ee1.sqrt();
            let f = (a * (1. - t) + b * t) / e.sqrt();
            (f, (b - a) / (ee1 - ee0) / e.sqrt() - f / (2. * e))
        };
        let (a, da) = energy_eval(im);
        let (b, db) = energy_eval(im + 1);
        let fm = if mu < self.mu[0] || mu > *self.mu.last().unwrap() {
            0.
        } else {
            (b - a) / (self.mu[im + 1] - self.mu[im])
        };
        (a * (1. - u) + b * u, da * (1. - u) + db * u, fm)
    }
    fn terms(&self, c: &Core, w: f64) -> (f64, f64) {
        let gyro = E * c.b / ME;
        let maxe = *self.energy.last().unwrap();
        let mut mh = 0.;
        let mut im = 0.;
        let nodes = [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ];
        let weights = [
            0.3478548451374539,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374539,
        ];
        let nearest = (w / gyro).round() as i32;
        let spread = ((c.kp * (2. * maxe * E / ME).sqrt() / gyro).ceil() as i32).max(1);
        for n in nearest - spread..=nearest + spread {
            let vpar = (w - n as f64 * gyro) / c.kp;
            let emin = ME * vpar * vpar / (2. * E);
            let lo = emin.max(self.energy[0]);
            if lo >= maxe {
                continue;
            }
            let mut breaks = vec![lo, maxe];
            breaks.extend(self.energy.iter().copied().filter(|&x| x > lo && x < maxe));
            for &mu in &self.mu {
                if mu * vpar > 0. {
                    let x = emin / (mu * mu);
                    if x > lo && x < maxe {
                        breaks.push(x);
                    }
                }
            }
            breaks.sort_by(f64::total_cmp);
            breaks.dedup();
            for win in breaks.windows(2) {
                let mid = (win[0] + win[1]) / 2.;
                let half = (win[1] - win[0]) / 2.;
                for (&x, &weight) in nodes.iter().zip(&weights) {
                    let ev = mid + half * x;
                    let v = (2. * ev * E / ME).sqrt();
                    let mu = vpar / v;
                    let (f, fe, fmu) = self.evaluate(ev, mu);
                    let aa = c.kt / gyro * (2. * E * (ev - emin).max(0.) / ME).sqrt();
                    let jj = libm::jn(n, aa).powi(2);
                    let dw = half * weight;
                    mh += dw * f * jj;
                    im += dw * (w * fe + E * (c.kp / (ME * v) - mu * w / (ME * v * v)) * fmu) * jj;
                }
            }
        }
        let mh = 2. * PI * PI * E / (c.ne * c.kp * ME) * mh;
        let im = -2. * PI * PI * E * E / (EPS * ME * c.k * c.k * c.kp) * im;
        (mh, im)
    }
}

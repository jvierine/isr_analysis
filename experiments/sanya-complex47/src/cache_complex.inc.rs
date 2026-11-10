struct HotCache {
    source_meta: serde_json::Value,
    height: Vec<f64>,
    values: Vec<f32>,
    du: f64,
    u0: f64,
    nu: usize,
    alpha: f64,
    real_multiplier: f64,
}
impl HotCache {
    fn load(path: &str) -> Result<Self> {
        let h = hdf5::File::open(path)?;
        let meta: serde_json::Value =
            serde_json::from_slice(&h.dataset("metadata_json_utf8")?.read_raw::<u8>()?)?;
        ensure!(
            meta["fields"][2] == "real_hot_susceptibility",
            "complex cache required"
        );
        Ok(Self {
            source_meta: meta["photoelectron_model"].clone(),
            height: h.dataset("height_km")?.read_raw::<f64>()?,
            values: h.dataset("hot_terms")?.read_raw::<f32>()?,
            du: meta["du"].as_f64().unwrap(),
            u0: meta["u0"].as_f64().unwrap(),
            nu: meta["nu"].as_u64().unwrap() as usize,
            alpha: meta["aspect_from_perpendicular_deg"].as_f64().unwrap(),
            real_multiplier: std::env::var("HOT_REAL_MULTIPLIER")
                .ok()
                .map(|s| s.parse().expect("invalid HOT_REAL_MULTIPLIER"))
                .unwrap_or(1.),
        })
    }
    fn at(&self, h: f64) -> HotSlice<'_> {
        assert!(h >= self.height[0] && h <= *self.height.last().unwrap());
        let j = self
            .height
            .partition_point(|&x| x < h)
            .saturating_sub(1)
            .min(self.height.len() - 2);
        HotSlice {
            cache: self,
            j,
            t: ((h - self.height[j]) / (self.height[j + 1] - self.height[j])).clamp(0., 1.),
        }
    }
}
struct HotSlice<'a> {
    cache: &'a HotCache,
    j: usize,
    t: f64,
}
impl HotSlice<'_> {
    fn response(&self, c: &Core, w: f64) -> (f64, C) {
        assert!(
            (c.kp / c.k - self.cache.alpha.to_radians().sin()).abs() < 1e-10,
            "wrong aspect for hot cache"
        );
        let q = (w / (E * c.b / ME) - self.cache.u0) / self.cache.du;
        assert!(
            q >= 0. && q < self.cache.nu as f64 - 1.,
            "frequency outside complex cache"
        );
        let i = q.floor() as usize;
        let u = q - i as f64;
        let at = |j: usize, k: usize| {
            let n = 3 * (j * self.cache.nu + i) + k;
            self.cache.values[n] as f64 * (1. - u) + self.cache.values[n + 3] as f64 * u
        };
        let v = |k| (at(self.j, k) * (1. - self.t) + at(self.j + 1, k) * self.t);
        (v(0) / c.ne, C::new(self.cache.real_multiplier * v(2), v(1)))
    }
}

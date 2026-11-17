//! Independent mean replay from previously saved native lag products and block backgrounds.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(
        a.len() == 3 || a.len() == 4,
        "validate-partials original-root partial-root [window]"
    );
    let mut max = 0f64;
    let mut pairs = 0usize;
    let windows = if a.len() == 4 {
        vec![a[3].parse::<usize>()?]
    } else {
        (0..48).collect()
    };
    let nw = windows.len();
    for t in windows {
        let old = hdf5::File::open(format!("{}/windows/{t:03}/lpm.h5", a[1]))?;
        let p = hdf5::File::open(format!("{}/windows/{t:03}/partial.h5", a[2]))?;
        let get = |name: &str| old.dataset(name).unwrap().read_raw::<f64>().unwrap();
        let re = get("native_lpm_real_adu2");
        let im = get("native_lpm_imag_adu2");
        let br = get("background_acf_real_adu2");
        let bi = get("background_acf_imag_adu2");
        let count = get("block_retained_pulses");
        let sum: f64 = count.iter().sum();
        let pr = p.dataset("acf_real_adu2")?.read_raw::<f64>()?;
        let pi = p.dataset("acf_imag_adu2")?.read_raw::<f64>()?;
        let sr = p.dataset("acf_real_sem_adu2")?.read_raw::<f64>()?;
        let si = p.dataset("acf_imag_sem_adu2")?.read_raw::<f64>()?;
        let indices = p.dataset("native_gate_index")?.read_raw::<u64>()?;
        for (g, &i) in indices.iter().enumerate() {
            let i = i as usize;
            for l in 0usize..61 {
                let bg: C = (0..count.len())
                    .filter(|&b| count[b] > 0.)
                    .map(|b| C::new(br[b * 61 + l], bi[b * 61 + l]) * count[b] / sum)
                    .sum();
                let first = p
                    .attr("first_stable_sample")
                    .ok()
                    .map(|a| a.read_scalar::<u64>().unwrap() as usize)
                    .unwrap_or(0);
                let lo = i.saturating_sub(l).max(first);
                let hi = i + 21;
                let native: C = (lo..hi)
                    .map(|k| C::new(re[l * 3782 + k], im[l * 3782 + k]))
                    .sum();
                let want = (native / (hi - lo) as f64 - bg) * 200. / (200 - l) as f64;
                let j = g * 61 + l;
                let err = ((want.re - pr[j]).abs() / sr[j].max(1.))
                    .max((want.im - pi[j]).abs() / si[j].max(1.));
                max = max.max(err);
                pairs += 1;
            }
        }
    }
    ensure!(max < 1e-5, "partial/native mismatch {max} SEM");
    let result = serde_json::json!({"windows":nw,"complex_products":pairs,"maximum_mean_discrepancy_in_SEM":max,"check":"independently recomposed from saved original native-lag means and pulse-count-weighted original background; all early range/lag products"});
    std::fs::write(
        format!("{}/partial-validation.json", a[2]),
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

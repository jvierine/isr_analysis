//! Weighted lag-profile inversion using f64 complex arithmetic. The unknowns
//! are narrow range gates and an optional range-independent background column.
use crate::{
    signal::{lag_products, lowpass, percentile},
    C,
};
use anyhow::{ensure, Result};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    /// Physical pulse energy in samples at unit peak transmit power, one entry
    /// per pulse. Supplied explicitly because pulse packs may be unit-energy.
    #[serde(default)]
    pub code_energy_samples: Vec<f64>,
    pub sample_rate_hz: f64,
    pub decimation: usize,
    pub lag_samples: Vec<usize>,
    #[serde(default)]
    pub minimum_range_index: Vec<usize>,
    #[serde(default)]
    pub science_pulse: Vec<bool>,
    pub range_start: usize,
    pub range_end: usize,
    pub measurement_start: usize,
    pub measurement_end: usize,
    pub tx_gate: [usize; 2],
    pub echo_gate: [usize; 2],
    pub noise_gate: [usize; 2],
    pub injection_gate: [usize; 2],
    pub filter_half: usize,
    pub passband_hz: f64,
    #[serde(default = "default_threshold")]
    pub outlier_threshold: f64,
    #[serde(default)]
    pub background_column: bool,
}
fn default_threshold() -> f64 {
    10.
}
pub struct Pulse {
    pub echo: Vec<C>,
    pub transmit: Vec<C>,
    pub pair: usize,
    pub usable: Vec<bool>,
}
/// Retained direct voltage-product equations, before any lag-profile solve.
#[derive(Clone,Debug,Serialize,Deserialize)]
pub struct DirectLagProducts {
    pub lag_samples: usize,
    pub range_columns: usize,
    pub background_column: bool,
    /// Actual retained pulse and decimated receive-bin identity.
    pub sample_identity: Vec<(usize,usize)>,
    pub rows: Vec<Vec<(usize,C)>>,
    pub products_kelvin: Vec<C>,
    /// Standard deviation of the complex product; circular-noise approximation.
    pub sigma_kelvin: Vec<f64>,
}
pub struct Estimate {
    pub direct_products: Vec<DirectLagProducts>,
    pub acf: Vec<C>,
    pub ground_clutter_acf: Vec<C>,
    pub variance: Vec<f64>,
    pub retained_fraction: Vec<f64>,
    pub condition_proxy: Vec<f64>,
    pub noise_power: f64,
    pub injection_power: f64,
    pub receiver_gain: f64,
    pub system_temperature_k: f64,
    /// Arithmetic mean of clean off-echo power / gain, for subtracting mean products.
    /// Gain calibration deliberately retains its existing median estimator.
    pub mean_receiver_background_k: f64,
    pub mean_receiver_background_se_k: f64,
    pub receiver_background_pulses: usize,
}
/// Recorded calibration failure, distinguishable from a numerical/layout error.
#[derive(Debug)]
pub struct InvalidReceiverGain {
    pub noise_power: f64,
    pub injection_power: f64,
    pub injection_temperature_k: f64,
    pub receiver_gain: f64,
}
impl std::fmt::Display for InvalidReceiverGain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "noise injection does not establish positive receiver gain (noise={}, injection={}, gain={})", self.noise_power, self.injection_power, self.receiver_gain)
    }
}
impl std::error::Error for InvalidReceiverGain {}
/// Column-scaled Hermitian WLS. Exactly unconstrained columns remain NaN;
/// non-positive-definite systems fail rather than silently regularizing.
pub fn solve(
    rows: &[Vec<(usize, C)>],
    y: &[C],
    sigma: &[f64],
    n: usize,
) -> Result<(Vec<C>, Vec<f64>, f64)> {
    ensure!(
        rows.len() == y.len() && y.len() == sigma.len() && n > 0,
        "WLS dimensions"
    );
    let mut scale = vec![0.; n];
    for (row, &s) in rows.iter().zip(sigma) {
        ensure!(s.is_finite() && s > 0., "nonpositive row sigma");
        for &(j, a) in row {
            ensure!(j < n, "column outside design");
            scale[j] += a.norm_sqr() / (s * s)
        }
    }
    let active: Vec<_> = (0..n).filter(|&j| scale[j] > 0.).collect();
    ensure!(!active.is_empty(), "unconstrained design");
    for s in &mut scale {
        *s = s.sqrt()
    }
    let mut map = vec![usize::MAX; n];
    for (i, &j) in active.iter().enumerate() {
        map[j] = i
    }
    let mut normal = DMatrix::<C>::zeros(active.len(), active.len());
    let mut rhs = DVector::<C>::zeros(active.len());
    for ((row, &y), &s) in rows.iter().zip(y).zip(sigma) {
        ensure!(
            y.re.is_finite() && y.im.is_finite(),
            "nonfinite measurement"
        );
        for &(j, a) in row {
            let a = a / (s * scale[j]);
            rhs[map[j]] += a.conj() * y / s;
            for &(k, b) in row {
                normal[(map[j], map[k])] += a.conj() * b / (s * scale[k]);
            }
        }
    }
    let factor = normal
        .cholesky()
        .ok_or_else(|| anyhow::anyhow!("rank-deficient lag design"))?;
    let diag: Vec<_> = factor.l().diagonal().iter().map(|v| v.norm_sqr()).collect();
    let ratio = diag.iter().copied().fold(f64::INFINITY, f64::min)
        / diag.iter().copied().fold(0., f64::max);
    ensure!(ratio > 1e-12, "ill-conditioned lag design");
    let answer = factor.solve(&rhs);
    let cov = factor.inverse();
    let mut x = vec![C::new(f64::NAN, f64::NAN); n];
    let mut variance = vec![f64::NAN; n];
    for (i, &j) in active.iter().enumerate() {
        x[j] = answer[i] / scale[j];
        variance[j] = cov[(i, i)].re / (scale[j] * scale[j])
    }
    Ok((x, variance, ratio))
}
fn mean_power(z: &[C], window: [usize; 2], dc: C) -> f64 {
    z[window[0]..window[1]]
        .iter()
        .map(|z| (*z - dc).norm_sqr())
        .sum::<f64>()
        / (window[1] - window[0]) as f64
}
fn receiver_calibration(pulses:&[Pulse],filtered:&[Vec<C>],s:&Settings,injection_temperature_k:f64)->Result<(f64,f64,f64)>{
    ensure!(injection_temperature_k.is_finite() && injection_temperature_k>0.,"invalid noise-injection temperature");
    // A detected echo or clipped sample must not contaminate calibration either.
    // Include the FIR input footprint of both calibration windows.
    let calibration:Vec<_>=filtered.iter().zip(pulses).filter(|(_,p)|{
        [s.noise_gate,s.injection_gate].iter().all(|g| {
            let lo=g[0].saturating_sub(s.filter_half);
            let hi=g[1].saturating_add(s.filter_half).min(p.usable.len());
            lo<hi && p.usable[lo..hi].iter().all(|v|*v)
        })
    }).map(|(z,_)|z).collect();
    ensure!(!calibration.is_empty(),"no usable receiver calibration pulses");
    let means: Vec<_> = calibration
        .iter()
        .map(|z| {
            z[s.noise_gate[0]..s.noise_gate[1]].iter().sum::<C>()
                / (s.noise_gate[1] - s.noise_gate[0]) as f64
        })
        .collect();
    let dc = C::new(
        percentile(&means.iter().map(|z| z.re).collect::<Vec<_>>(), 0.5),
        percentile(&means.iter().map(|z| z.im).collect::<Vec<_>>(), 0.5),
    );
    let noise = percentile(
        &calibration
            .iter()
            .map(|z| mean_power(z, s.noise_gate, dc))
            .collect::<Vec<_>>(),
        0.5,
    );
    let injection = percentile(
        &calibration
            .iter()
            .map(|z| mean_power(z, s.injection_gate, dc))
            .collect::<Vec<_>>(),
        0.5,
    );
    let gain = (injection - noise) / injection_temperature_k;
    if !gain.is_finite() || gain <= 0. {
        return Err(InvalidReceiverGain {noise_power:noise,injection_power:injection,injection_temperature_k,receiver_gain:gain}.into());
    }
    Ok((noise,injection,gain))
}
fn mean_receiver_background(pulses:&[Pulse],filtered:&[Vec<C>],s:&Settings)->Result<(f64,f64,usize)>{
    let powers:Vec<f64>=filtered.iter().zip(pulses).filter(|(_,p)| {
        [s.noise_gate,s.injection_gate].iter().all(|g| {
            let lo=g[0].saturating_sub(s.filter_half);
            let hi=g[1].saturating_add(s.filter_half).min(p.usable.len());
            lo<hi && p.usable[lo..hi].iter().all(|v|*v)
        })
    }).map(|(z,_)|z[s.noise_gate[0]..s.noise_gate[1]].iter().map(|v|v.norm_sqr()).sum::<f64>()/(s.noise_gate[1]-s.noise_gate[0])as f64).collect();
    ensure!(!powers.is_empty() && powers.iter().all(|x|x.is_finite()),"insufficient clean receiver-background measurements");
    let mean=powers.iter().sum::<f64>()/powers.len()as f64;
    let se=if powers.len()>1 {(powers.iter().map(|x|(x-mean).powi(2)).sum::<f64>()/((powers.len()-1)*powers.len())as f64).sqrt()}else{f64::NAN};
    Ok((mean,se,powers.len()))
}
pub fn estimate(pulses: &[Pulse], s: &Settings, injection_temperature_k: f64) -> Result<Estimate> {
    estimate_with_progress(pulses, s, injection_temperature_k, |_, _, _| Ok(()))
}
pub fn estimate_with_progress(
    pulses: &[Pulse],
    s: &Settings,
    injection_temperature_k: f64,
    progress: impl FnMut(usize, usize, &Estimate) -> Result<()>,
) -> Result<Estimate> {
    estimate_impl(pulses,s,injection_temperature_k,false,None,progress)
}
/// Extract direct lagged products with the same masks, calibration and weights.
/// No call to the lag-profile solver is made in this mode.
pub fn direct_products(pulses:&[Pulse],s:&Settings,injection_temperature_k:f64)->Result<Estimate>{
    estimate_impl(pulses,s,injection_temperature_k,true,None,|_,_,_|Ok(()))
}
pub fn direct_products_shrunk_variance(pulses:&[Pulse],s:&Settings,t:f64,codes:&[u64])->Result<Estimate>{
 ensure!(codes.len()==pulses.len(),"missing pulse code identities");estimate_impl(pulses,s,t,true,Some((codes,true)),|_,_,_|Ok(()))
}
/// Same-code, same-bin centered complex sample variance; per-component sigma.
pub fn direct_products_sample_variance(pulses:&[Pulse],s:&Settings,t:f64,codes:&[u64])->Result<Estimate>{
 ensure!(codes.len()==pulses.len(),"missing pulse code identities for sample variance");
 estimate_impl(pulses,s,t,true,Some((codes,false)),|_,_,_|Ok(()))
}
#[derive(Clone,Default)]
struct SampleVariance { n:usize, mean:C, m2:f64 }
impl SampleVariance {
 fn shrunk_sigma(&self,pool:f64)->f64{let sigma=self.sigma();if sigma.is_finite()&&pool.is_finite()&&pool>0.{(((self.n-1)as f64*sigma.powi(2)+8.*pool)/((self.n-1)as f64+8.)).sqrt()}else{f64::NAN}}

 fn add(&mut self,z:C){self.n+=1;let delta=z-self.mean;self.mean+=delta/self.n as f64;self.m2+=(delta.conj()*(z-self.mean)).re;}
 fn sigma(&self)->f64 {if self.n>=4 && self.m2>0. {(self.m2/(2.*(self.n-1)as f64)).sqrt()}else{f64::NAN}}
}
fn estimate_impl(
    pulses:&[Pulse],s:&Settings,injection_temperature_k:f64,direct_only:bool,sample_codes:Option<(&[u64],bool)>,
    mut progress:impl FnMut(usize,usize,&Estimate)->Result<()>,
)->Result<Estimate>{
    ensure!(
        pulses.len() >= 16
            && s.decimation > 0
            && s.range_end > s.range_start
            && s.measurement_end > s.measurement_start
            && !s.lag_samples.is_empty(),
        "insufficient pulses or invalid LPI geometry"
    );
    let n = pulses[0].echo.len();
    ensure!(s.code_energy_samples.is_empty() || (s.code_energy_samples.len()==pulses.len() && s.code_energy_samples.iter().all(|e|e.is_finite()&&*e>0.)), "one positive physical pulse energy is required per pulse");
    let range_cell_km=s.decimation as f64/s.sample_rate_hz*299792.458/2.;
    ensure!(
        pulses.iter().all(|p| p.echo.len() == n
            && p.transmit.len() == n
            && p.usable.len() == n
            && p.pair < pulses.len()),
        "pulse dimensions/pair index mismatch"
    );
    ensure!(
        s.lag_samples.iter().all(|&l| l < n)
            && [s.tx_gate, s.echo_gate, s.noise_gate, s.injection_gate]
                .iter()
                .all(|g| g[0] < g[1] && g[1] <= n)
            && s.measurement_end * s.decimation + s.lag_samples.iter().max().unwrap() <= n,
        "invalid gate timing"
    );
    ensure!(s.science_pulse.is_empty()||s.science_pulse.len()==pulses.len(),"science pulse mask length mismatch");
    ensure!(s.minimum_range_index.is_empty()||s.minimum_range_index.len()==s.lag_samples.len(),"minimum range index length mismatch");
    let nr = s.range_end - s.range_start;
    let nl = s.lag_samples.len();
    let cols = nr + usize::from(s.background_column);
    let mut filtered = Vec::new();
    let mut txs = Vec::new();
    let mut masks = Vec::new();
    for p in pulses {
        filtered.push(lowpass(
            &p.echo,
            s.sample_rate_hz,
            1.2 * s.passband_hz,
            s.filter_half,
        )?);
        let mut tx = p.transmit.clone();
        for (i, z) in tx.iter_mut().enumerate() {
            if !(s.tx_gate[0]..s.tx_gate[1]).contains(&i) {
                *z = C::default()
            }
        }
        let norm = tx.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        ensure!(norm > 0., "zero transmit energy");
        for z in &mut tx {
            *z /= norm
        }
        txs.push(tx);
        let mut mask = p.usable.clone();
        for i in 0..n {
            if !p.usable[i] {
                for m in mask
                    .iter_mut()
                    .take((i + s.filter_half + 1).min(n))
                    .skip(i.saturating_sub(s.filter_half))
                {
                    *m = false
                }
            }
        }
        for (i, m) in mask.iter_mut().enumerate() {
            if !(s.echo_gate[0] + s.filter_half..s.echo_gate[1].saturating_sub(s.filter_half))
                .contains(&i)
            {
                *m = false
            }
        }
        masks.push(mask);
    }
    let (noise,injection,gain)=receiver_calibration(pulses,&filtered,s,injection_temperature_k)?;
    let (background,background_se,background_pulses)=mean_receiver_background(pulses,&filtered,s)?;
    let mut out = Estimate {
        direct_products: vec![],
        acf: vec![C::new(f64::NAN, f64::NAN); nr * nl],
        ground_clutter_acf: vec![C::new(f64::NAN, f64::NAN); nr * nl],
        variance: vec![f64::NAN; nr * nl],
        retained_fraction: vec![0.; nl],
        condition_proxy: vec![0.; nl],
        noise_power: noise,
        injection_power: injection,
        receiver_gain: gain,
        system_temperature_k: noise / gain,
        mean_receiver_background_k: background/gain,
        mean_receiver_background_se_k: background_se/gain,
        receiver_background_pulses: background_pulses,
    };
    for (li, &lag) in s.lag_samples.iter().enumerate() {
        let mut products = Vec::new();
        let mut difference = Vec::new();
        let mut ambiguities = Vec::new();
        for (pi, p) in pulses.iter().enumerate() {
            products.push(lag_products(&filtered[pi], lag, s.decimation)?);
            let zd: Vec<_> = filtered[pi]
                .iter()
                .zip(&filtered[p.pair])
                .map(|(a, b)| a - b)
                .collect();
            difference.push(lag_products(&zd, lag, s.decimation)?);
            let mut ambiguity=lag_products(&txs[pi], lag, s.decimation)?;
            // Restore physical unit-peak pulse energy removed by the TX
            // normalization, and integrate the range basis in kilometres.
            // Scaling each pulse's design permits mixed baud lengths; dividing
            // the final average by one nominal baud does not.
            if !s.code_energy_samples.is_empty(){for a in &mut ambiguity{*a*=s.code_energy_samples[pi]*range_cell_km;}}
            ambiguities.push(ambiguity);
        }
        let valid_product=|pi:usize,m:usize| {
            let pair=pulses[pi].pair;
            (m*s.decimation..(m+1)*s.decimation).all(|i|masks[pi][i]&&masks[pi][i+lag]&&masks[pair][i]&&masks[pair][i+lag])
        };
        let mut empirical=std::collections::BTreeMap::<(u64,usize),SampleVariance>::new();
        if let Some((codes,_))=sample_codes {
            for pi in 0..pulses.len(){
                if !s.science_pulse.is_empty()&&!s.science_pulse[pi]{continue;}
                for m in s.measurement_start..s.measurement_end {
                    if valid_product(pi,m){empirical.entry((codes[pi],m)).or_default().add(products[pi][m]);}
                }
            }
        }
        let nm = s.measurement_end - s.measurement_start;
        let pooled:Vec<f64>=(s.measurement_start..s.measurement_end).map(|m|{
            let values:Vec<_>=empirical.iter().filter(|((_,bin),v)|*bin==m&&v.n>=4).map(|(_,v)|v.sigma().powi(2)).filter(|x|x.is_finite()&&*x>0.).collect();
            if values.is_empty(){f64::NAN}else{percentile(&values,0.5)}
        }).collect();
        let robust: Vec<_> = (s.measurement_start..s.measurement_end)
            .map(|m| {
                (2. * percentile(
                    &products.iter().map(|p| p[m].norm_sqr()).collect::<Vec<_>>(),
                    0.34,
                ))
                .sqrt()
            })
            .collect();
        let robust_g: Vec<_> = (s.measurement_start..s.measurement_end)
            .map(|m| {
                (2. * percentile(
                    &difference
                        .iter()
                        .map(|p| p[m].norm_sqr())
                        .collect::<Vec<_>>(),
                    0.34,
                ))
                .sqrt()
            })
            .collect();
        // Direct f64 moving power sum: never FFT a single-precision weight matrix
        // dominated by a bright satellite. Preserve the legacy ten-pulse window.
        let local: Vec<Vec<f64>> = (0..pulses.len())
            .map(|p| {
                (s.measurement_start..s.measurement_end)
                    .map(|m| {
                        ((0..10)
                            .map(|j| {
                                products[(p + pulses.len() + 5 - j % pulses.len()) % pulses.len()]
                                    [m]
                                    .norm_sqr()
                            })
                            .sum::<f64>()
                            / 10.)
                            .sqrt()
                    })
                    .collect()
            })
            .collect();
        let median = percentile(&local.iter().flatten().copied().collect::<Vec<_>>(), 0.5);
        ensure!(
            median > 0. && median.is_finite(),
            "invalid product noise weights"
        );
        let mut identities = Vec::new();
        let mut rows = Vec::new();
        let mut y = Vec::new();
        let mut yg = Vec::new();
        let mut sigma = Vec::new();
        for (pi, p) in pulses.iter().enumerate() {
            if !s.science_pulse.is_empty()&&!s.science_pulse[pi]{continue;}
            for (mi, m) in (s.measurement_start..s.measurement_end).enumerate() {
                let start = m * s.decimation;
                let end = (m + 1) * s.decimation;
                let valid = (start..end).all(|i| {
                    masks[pi][i] && masks[pi][i + lag] && masks[p.pair][i] && masks[p.pair][i + lag]
                });
                let weight = if let Some((codes,shrink))=sample_codes {empirical.get(&(codes[pi],m)).map_or(f64::NAN,|v|if shrink {v.shrunk_sigma(pooled[mi])}else{v.sigma()})}else{local[pi][mi].max(median)};
                if !valid || !weight.is_finite() || weight<=0.
                    || products[pi][m].norm() > s.outlier_threshold * robust[mi]
                    || difference[pi][m].norm() > s.outlier_threshold * robust_g[mi]
                    || (sample_codes.is_none() && weight > 100. * median)
                {
                    continue;
                }
                let mut row = Vec::new();
                for ri in s.range_start.max(s.minimum_range_index.get(li).copied().unwrap_or(s.range_start))..s.range_end {
                    if m >= ri {
                        let j = m - ri;
                        if j < ambiguities[pi].len() {
                            let a = ambiguities[pi][j];
                            if a.norm_sqr() > 1e-30 {
                                row.push((ri - s.range_start, a))
                            }
                        }
                    }
                }
                if s.background_column {
                    row.push((nr, C::new(1., 0.)))
                }
                if row.is_empty() {
                    continue;
                }
                identities.push((pi,m));
                rows.push(row);
                y.push(products[pi][m]);
                yg.push(difference[pi][m]);
                sigma.push(weight);
            }
        }
        out.retained_fraction[li] = rows.len() as f64 / (pulses.len() * nm) as f64;
        if direct_only {
            out.direct_products.push(DirectLagProducts {
                lag_samples:lag,range_columns:nr,background_column:s.background_column,
                sample_identity:identities,rows,products_kelvin:y.iter().map(|v|*v/gain).collect(),
                sigma_kelvin:sigma.iter().map(|v|*v/gain).collect(),
            });
            progress(li+1,nl,&out)?;
            continue;
        }
        if rows.len() < cols {
            continue;
        }
        let result = solve(&rows, &y, &sigma, cols);
        let result_g = solve(&rows, &yg, &sigma, cols);
        if let (Ok((x, var, condition)), Ok((xg, _, _))) = (result, result_g) {
            out.condition_proxy[li] = condition;
            for ri in 0..nr {
                out.acf[ri * nl + li] = x[ri] / gain;
                out.ground_clutter_acf[ri * nl + li] = xg[ri] / (2. * gain);
                out.variance[ri * nl + li] = var[ri] / gain.powi(2);
            }
        }
        progress(li + 1, nl, &out)?;
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scaled_wls_and_unconstrained() {
        let rows = vec![
            vec![(0, C::new(1e-9, 0.)), (1, C::new(0., 1e8))],
            vec![(0, C::new(2e-9, 0.)), (1, C::new(1e8, 0.))],
            vec![(0, C::new(3e-9, 0.)), (1, C::new(-1e8, 0.))],
        ];
        let truth = [C::new(2., -3.), C::new(4., 1.)];
        let y: Vec<_> = rows
            .iter()
            .map(|r| r.iter().map(|(j, a)| truth[*j] * a).sum())
            .collect();
        let (x, v, _) = solve(&rows, &y, &[1.; 3], 3).unwrap();
        assert!((x[1] - truth[1]).norm() < 1e-12);
        assert!(x[2].re.is_nan() && v[2].is_nan());
    }
    #[test]
    fn singular_design_is_rejected() {
        let rows = vec![vec![(0, C::new(1., 0.)), (1, C::new(1., 0.))]; 4];
        assert!(solve(&rows, &[C::new(1., 0.); 4], &[1.; 4], 2).is_err());
    }
    #[test]
    fn mean_background_matches_mean_products_and_respects_masks(){
        let s:Settings=serde_json::from_value(serde_json::json!({"sample_rate_hz":1.,"decimation":1,"lag_samples":[1],"range_start":0,"range_end":1,"measurement_start":0,"measurement_end":1,"tx_gate":[0,1],"echo_gate":[0,4],"noise_gate":[0,2],"injection_gate":[2,4],"filter_half":1,"passband_hz":0.1})).unwrap();
        let filtered=vec![vec![C::new(1.,0.);4],vec![C::new(1.,0.);4],vec![C::new(2.,0.);4]];
        let mut pulses:Vec<_>=(0..3).map(|_|Pulse{echo:vec![],transmit:vec![],pair:0,usable:vec![true;4]}).collect();
        let (mean,se,n)=mean_receiver_background(&pulses,&filtered,&s).unwrap();
        assert_eq!(n,3);assert!((mean-2.).abs()<1e-14);assert!((se-1.).abs()<1e-14);
        pulses[2].usable[2]=false;
        assert_eq!(mean_receiver_background(&pulses,&filtered,&s).unwrap(),(1.,0.,2));
    }
    #[test]
    fn receiver_gain_excludes_fully_rejected_pulses(){
        let s:Settings=serde_json::from_value(serde_json::json!({"sample_rate_hz":1.,"decimation":1,"lag_samples":[1],"range_start":0,"range_end":1,"measurement_start":0,"measurement_end":1,"tx_gate":[0,1],"echo_gate":[0,4],"noise_gate":[0,2],"injection_gate":[2,4],"filter_half":1,"passband_hz":0.1})).unwrap();
        let mut filtered=Vec::new();let mut pulses=Vec::new();
        for i in 0..56{let a=if i<16{1.}else{1e9};filtered.push(vec![C::new(-a,0.),C::new(a,0.),C::new(-5f64.sqrt()*a,0.),C::new(5f64.sqrt()*a,0.)]);pulses.push(Pulse{echo:vec![],transmit:vec![],pair:0,usable:vec![i<16;4]});}
        let(noise,injection,gain)=receiver_calibration(&pulses,&filtered,&s,2.).unwrap();assert!((noise-1.).abs()<1e-14);assert!((injection-5.).abs()<1e-14);assert!((gain-2.).abs()<1e-14);
        // Even a partially masked calibration window invalidates that pulse's gain sample.
        for p in &mut pulses{p.usable[1]=false;}
        assert!(receiver_calibration(&pulses,&filtered,&s,2.).is_err());
    }
    #[test]
    fn invalid_gain_retains_measurements_as_a_typed_quality_failure(){
        let s:Settings=serde_json::from_value(serde_json::json!({"sample_rate_hz":1.,"decimation":1,"lag_samples":[1],"range_start":0,"range_end":1,"measurement_start":0,"measurement_end":1,"tx_gate":[0,1],"echo_gate":[0,4],"noise_gate":[0,2],"injection_gate":[2,4],"filter_half":1,"passband_hz":0.1})).unwrap();
        let filtered=vec![vec![C::new(-2.,0.),C::new(2.,0.),C::new(-1.,0.),C::new(1.,0.)]];
        let pulses=vec![Pulse{echo:vec![],transmit:vec![],pair:0,usable:vec![true;4]}];
        let error=receiver_calibration(&pulses,&filtered,&s,2.).unwrap_err();
        let q=error.downcast_ref::<InvalidReceiverGain>().expect("calibration quality error");
        assert_eq!((q.noise_power,q.injection_power,q.receiver_gain),(4.,1.,-1.5));
        let error=receiver_calibration(&pulses,&filtered,&s,0.).unwrap_err();
        assert!(error.downcast_ref::<InvalidReceiverGain>().is_none(),"a bad configuration must still fail");
    }

}

#[cfg(test)] mod sample_variance_tests {
 use super::*;
 #[test] fn pooling_stabilizes_small_sample_variance_without_changing_mean(){
  let mut a=SampleVariance::default();for x in [-0.2,-0.1,0.1,0.2]{a.add(C::new(x,0.));}
  let empirical=a.sigma().powi(2);let pooled=a.shrunk_sigma(2.).powi(2);
  assert!((pooled-(3.*empirical+16.)/11.).abs()<1e-12);assert!(pooled>empirical&&pooled<2.);
  assert!(a.mean.norm()<1e-15);assert!(!a.shrunk_sigma(f64::NAN).is_finite());
 }
 #[test] fn centered_variance_ignores_signal_offset(){
  let samples=[C::new(1.,2.),C::new(-1.,0.),C::new(2.,-2.),C::new(-2.,0.)];
  let mut a=SampleVariance::default();let mut b=SampleVariance::default();
  for z in samples {a.add(z);b.add(z+C::new(10000.,-20000.));}
  assert!((a.sigma().powi(2)-3.).abs()<1e-12);assert!((a.sigma()-b.sigma()).abs()<1e-10);
  let mut few=SampleVariance::default();few.add(samples[0]);assert!(!few.sigma().is_finite());
 }
}

//! Paired same-process derivative benchmark on the actual native AC response.
use anyhow::{ensure, Result};
use std::{hint::black_box, path::Path, time::Instant};
use syisr_ac::{
    campaign::{self, Campaign},
    coupled::encode,
    decode, fullprofile, h_transition,
};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "INPUT_ROOT JOB_ID OUTPUT_HDF");
    let root = Path::new(&a[1]);
    let c: Campaign = campaign::load(&root.join("campaign.h5"))?;
    let j = c
        .jobs
        .iter()
        .find(|j| j.id == a[2])
        .ok_or_else(|| anyhow::anyhow!("unknown job"))?;
    let codes = decode::phase(&root.join("m32-reference.bc"))?;
    let threads = std::env::var("RAYON_NUM_THREADS")
        .unwrap_or("4".into())
        .parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let p = fullprofile::prepare(&c, j, root, &codes)?;
    let mut x = encode(&p.initial);
    let h = h_transition::Shape::encode(1000., 150.);
    let he = h_transition::helium_encode(0.1, 850., 150.);
    for (i, k) in [4, 10, 17, 23, 29].iter().enumerate() {
        x[*k] = if i < 2 { h[i] } else { he[i - 2] };
    }
    let (r, jac) = p.problem.evaluate_mode(&x, true, false)?;
    let (ar, aj) = p.problem.evaluate_autodiff(&x, true)?;
    let re = (&ar - &r).norm() / r.norm().max(1e-30);
    let je = (&aj - &jac).norm() / jac.norm().max(1e-30);
    ensure!(
        re < 1e-11 && je < 1e-3,
        "AD validation residual={re} jacobian={je}"
    );
    let mut column_errors = vec![];
    for k in 0..x.len() {
        let d = (aj.column(k) - jac.column(k)).norm();
        let scale = jac.column(k).norm();
        if scale > 1e-10 {
            column_errors.push((k, d / scale));
        }
    }
    let max_column = column_errors.iter().map(|(_, e)| *e).fold(0., f64::max);
    ensure!(max_column < 0.02, "unstable AD column {max_column}");
    drop((r, jac, ar, aj));
    let mut trials = vec![];
    for trial in 0..6 {
        for ad in if trial % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let t = Instant::now();
            let (r, j) = if ad {
                p.problem.evaluate_autodiff(&x, true)?
            } else {
                p.problem.evaluate_mode(&x, true, false)?
            };
            black_box(r.norm_squared() + j[(0, 0)]);
            trials.push(
                serde_json::json!({"trial":trial,"ad":ad,"wall_s":t.elapsed().as_secs_f64()}),
            );
        }
    }
    let result = serde_json::json!({"schema":"syisr-ad-benchmark-v1","job":j,"rayon_threads":threads,"source_cells":p.problem.cells.len(),"products":p.problem.products.len(),"parameter_storage_columns":x.len(),"residual_relative_error":re,"jacobian_relative_error":je,"max_column_relative_error":max_column,"column_errors":column_errors,"trials":trials,"cpu_info":std::fs::read_to_string("/proc/cpuinfo").ok(),"scope":"Full native response plus residual/Jacobian evaluation; not a full optimizer speedup. Alternating order, common setup and physical model; current host load applies."});
    campaign::save(Path::new(&a[3]), &result)?;
    println!(
        "{}",
        serde_json::json!({"residual_relative_error":re,"jacobian_relative_error":je,"max_column_relative_error":max_column,"trials":result["trials"],"products":p.problem.products.len(),"source_cells":p.problem.cells.len()})
    );
    Ok(())
}

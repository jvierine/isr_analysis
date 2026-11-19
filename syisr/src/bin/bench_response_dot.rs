//! Paired streaming-kernel benchmark; setup and allocation are outside timing.
use serde_json::json;
use std::{hint::black_box, time::Instant};
use syisr_ac::{campaign, response_dot, Complex as C};
fn main() -> anyhow::Result<()> {
    let a: Vec<_> = std::env::args().collect();
    let nf = 201;
    let rows = 32768;
    let percell = 1024;
    let channels: Vec<Vec<f64>> = (0..6)
        .map(|c| (0..nf).map(|i| ((i + c) as f64 * 0.137).cos()).collect())
        .collect();
    let kernel: Vec<C> = (0..nf * rows)
        .map(|i| C::new((i as f64 * 0.017).sin(), (i as f64 * 0.023).cos()))
        .collect();
    let mut output = vec![[C::default(); 6]; rows];
    let mut trials = vec![];
    for trial in 0..10 {
        for which in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
            let now = Instant::now();
            if which == 0 {
                for (cell, block) in kernel.chunks(nf * percell).enumerate() {
                    for (c, s) in channels.iter().enumerate() {
                        for (r, row) in block.chunks_exact(nf).enumerate() {
                            output[cell * percell + r][c] = black_box(row)
                                .iter()
                                .zip(black_box(s))
                                .map(|(k, s)| *k * *s)
                                .sum();
                        }
                    }
                }
            } else {
                for (row, dst) in kernel.chunks_exact(nf).zip(output.iter_mut()) {
                    response_dot::fused_into(black_box(row), black_box(&channels), black_box(dst));
                }
            }
            black_box(&output);
            let sec = now.elapsed().as_secs_f64();
            let checksum: C = output.iter().map(|v| v[0]).sum();
            if trial > 1 {
                trials.push(json!({"trial":trial,"implementation":if which==0{"previous_repeated_pass"}else{"fused"},"wall_s":sec,"checksum":[checksum.re,checksum.im]}));
            }
        }
    }
    campaign::save(
        std::path::Path::new(&a[1]),
        &json!({"scope":"streaming kernel only, no whole-fit speedup claim","nf":nf,"rows":rows,"per_cell_rows":percell,"channels":channels.len(),"kernel_bytes":kernel.len()*16,"arch":std::env::consts::ARCH,"cpu":std::env::var("BENCH_CPU").ok(),"threads":1,"trials":trials}),
    )?;
    Ok(())
}

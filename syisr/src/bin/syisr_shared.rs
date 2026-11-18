//! One MPI task per elapsed window. Checkpoints survive loss of the SSH client.
use anyhow::{ensure, Result};
use std::{collections::BTreeMap, fs, path::Path, sync::Arc};
use syisr_ac::{
    campaign::{self, Campaign, Job},
    coupled::{self, Fit},
    decode, fullprofile,
    h_transition::{HeliumLayer, Shape, Transition},
    sharedcomposition::{self, Temporal, TimeNeighbor, SHARED},
};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 6,
        "INPUT_ROOT WINDOW-or-all OUTPUT_ROOT PREVIOUS_STAGE-or-none MAX_ITER"
    );
    let input = Path::new(&a[1]);
    let out = Path::new(&a[3]);
    let previous = if a[4] == "none" {
        None
    } else {
        Some(Path::new(&a[4]))
    };
    let maxiter = a[5].parse()?;
    ensure!(
        std::env::var("SYISR_H_MODEL").ok().as_deref() == Some("iri_sigmoid"),
        "shared fit requires sigmoid + He envelope"
    );
    let threads = std::env::var("RAYON_NUM_THREADS")
        .unwrap_or("5".into())
        .parse()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let rank = std::env::var("OMPI_COMM_WORLD_RANK")
        .unwrap_or("0".into())
        .parse::<usize>()?;
    let size = std::env::var("OMPI_COMM_WORLD_SIZE")
        .unwrap_or("1".into())
        .parse::<usize>()?;
    let c: Campaign = campaign::load(&input.join("campaign.h5"))?;
    let codes = decode::phase(&input.join("m32-reference.bc"))?;
    let mut windows = BTreeMap::<i64, Vec<Job>>::new();
    for j in &c.jobs {
        windows.entry(j.start as i64).or_default().push(j.clone());
    }
    let times: Vec<_> = windows.keys().copied().collect();
    let mut failures = 0;
    for (wi, (&time, jobs)) in windows.iter().enumerate() {
        if (a[2] != "all" && a[2] != time.to_string()) || wi % size != rank {
            continue;
        }
        if out
            .join("shared-windows")
            .join(format!("{time}.h5"))
            .exists()
        {
            continue;
        }
        let result = (|| -> Result<()> {
            let mut prepared = vec![];
            let mut selected = vec![];
            let mut skipped = vec![];
            for j in jobs {
                match fullprofile::prepare(&c, j, input, &codes) {
                    Ok(p) => {
                        selected.push(j.clone());
                        prepared.push(p);
                    }
                    Err(e) => {
                        skipped.push(serde_json::json!({"job":j.id,"reason":format!("{e:#}")}));
                    }
                }
            }
            ensure!(!prepared.is_empty(), "no supported beams: {skipped:?}");
            let reference = selected.iter().position(|j| j.el > 89.).unwrap_or(0);
            let shape: Arc<Shape> = prepared[reference].shape.clone().unwrap();
            // One fixed shape for every beam is essential: equal parameters alone do not give equal fractions with beam-dependent shapes.
            for p in &mut prepared {
                p.shape = Some(shape.clone());
                for cell in &mut p.problem.cells {
                    if let Some((s, _)) = &mut cell.composition {
                        *s = shape.clone();
                    }
                }
            }
            let alt: Vec<_> = (600..=1300).step_by(100).map(|z| z as f64).collect();
            let mut neighbors = vec![];
            if let Some(previous) = previous {
                for n in [
                    wi.checked_sub(1),
                    if wi + 1 < times.len() {
                        Some(wi + 1)
                    } else {
                        None
                    },
                ]
                .into_iter()
                .flatten()
                {
                    let path = previous
                        .join("shared-windows")
                        .join(format!("{}.h5", times[n]));
                    ensure!(
                        path.exists(),
                        "previous sweep is incomplete at {}",
                        times[n]
                    );
                    let v: serde_json::Value = campaign::load(&path)?;
                    let tr: Transition = serde_json::from_value(v["h_transition"].clone())?;
                    neighbors.push(TimeNeighbor {
                        separation_s: (time - times[n]) as f64,
                        h_fraction: alt
                            .iter()
                            .map(|&z| tr.shape.evaluate(z, tr.normalized_parameters).0)
                            .collect(),
                    });
                }
            }
            let temporal = Temporal {
                altitude_km: alt,
                sigma_per_30min: 0.15,
                neighbors,
            };
            let mut initial: Vec<_> = prepared.iter().map(|p| p.initial.clone()).collect();
            // Exact previous transformed controls are a warm start only, never substituted observations.
            let seedroot = previous.map(|p| p.join("joint-fits")).or_else(|| {
                std::env::var_os("SYISR_JOINT_SEED_ROOT").map(std::path::PathBuf::from)
            });
            for (j, nodes) in selected.iter().zip(&mut initial) {
                if let Some(root) = &seedroot {
                    if let Ok(v) =
                        campaign::load::<serde_json::Value>(&root.join(format!("{}.h5", j.id)))
                    {
                        if let Some(ctrl) = v["control_nodes"]
                            .as_array()
                            .filter(|v| v.len() == nodes.len())
                        {
                            for (n, v) in nodes.iter_mut().zip(ctrl) {
                                n.power_adu2 = v["ne_m3"].as_f64().unwrap() / 1e11;
                                n.ti_k = v["ti_k"].as_f64().unwrap();
                                n.te_k = v["te_k"].as_f64().unwrap();
                                n.velocity_m_s = v["velocity_m_s"].as_f64().unwrap();
                            }
                            if let (Some(h), Some(he)) = (
                                v["h_transition"]["normalized_parameters"].as_array(),
                                v["helium_layer"]["normalized_parameters"].as_array(),
                            ) {
                                let q = [
                                    h[0].as_f64().unwrap(),
                                    h[1].as_f64().unwrap(),
                                    he[0].as_f64().unwrap(),
                                    he[1].as_f64().unwrap(),
                                    he[2].as_f64().unwrap(),
                                ];
                                let mut x = coupled::encode(nodes);
                                for (i, k) in SHARED.iter().enumerate() {
                                    x[*k] = q[i];
                                }
                                for (k, n) in nodes.iter_mut().enumerate() {
                                    n.h_fraction = x[6 * k + 4];
                                    n.he_fraction = (1. - n.h_fraction) * x[6 * k + 5];
                                    n.o_fraction = 1. - n.h_fraction - n.he_fraction;
                                }
                            }
                        }
                    }
                }
            }
            // Start the shared coordinates from the zenith reference, not the first beam in directory order.
            let seed = coupled::encode(&initial[reference]);
            for nodes in &mut initial {
                for k in 0..nodes.len() {
                    let h = if k == 0 {
                        seed[4]
                    } else if k == 1 {
                        seed[10]
                    } else {
                        0.
                    };
                    let he = if (2..5).contains(&k) {
                        seed[6 * k + 5]
                    } else {
                        0.
                    };
                    nodes[k].h_fraction = h;
                    nodes[k].he_fraction = (1. - h) * he;
                    nodes[k].o_fraction = 1. - h - nodes[k].he_fraction;
                }
            }
            eprintln!(
                "SHARED_START {time} beams={} skipped={} temporal_neighbors={}",
                prepared.len(),
                skipped.len(),
                temporal.neighbors.len()
            );
            let problems: Vec<_> = prepared.iter().map(|p| &p.problem).collect();
            let solution = sharedcomposition::solve(
                &problems,
                &initial,
                shape.clone(),
                Some(&temporal),
                maxiter,
            )?;
            let mut beam_stats = vec![];
            for ((p, j), nodes) in prepared.iter().zip(&selected).zip(&solution.nodes) {
                let x = coupled::encode(nodes);
                let (r, _) = p.problem.evaluate(&x, false)?;
                let chi = r.norm_squared();
                let nfree = 4 * p.problem.knot_count;
                let dof = (p
                    .problem
                    .products
                    .iter()
                    .map(|p| p.sigma.iter().filter(|&&s| s < 1e20).count())
                    .sum::<usize>())
                .saturating_sub(nfree);
                let predicted = p
                    .problem
                    .products
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        [
                            r[2 * i] * p.sigma[0] + p.value[0],
                            r[2 * i + 1] * p.sigma[1] + p.value[1],
                        ]
                    })
                    .collect();
                let fit = Fit {
                    status: solution.status.clone(),
                    iterations: solution.iterations,
                    chi_square: chi,
                    degrees_of_freedom: dof,
                    reduced_chi_square: chi / dof.max(1) as f64,
                    nodes: nodes.clone(),
                    conditional_standard_error: vec![],
                    transformed_covariance: None,
                    predicted,
                    standardized_residual: r.as_slice()[..2 * p.problem.products.len()]
                        .chunks_exact(2)
                        .map(|r| [r[0], r[1]])
                        .collect(),
                    objective_history: solution.objective_history.clone(),
                    active_bounds: vec![],
                    normalized_gradient: solution.normalized_gradient,
                    uncertainty: "Disabled".into(),
                };
                let mut joint = fullprofile::report_shared(p, j, fit)?;
                let measured: Vec<f64> = p
                    .problem
                    .products
                    .iter()
                    .enumerate()
                    .flat_map(|(i, p)| {
                        (0..2).filter_map({
                            let r = &r;
                            move |k| {
                                if p.sigma[k] < 1e20 {
                                    Some(r[2 * i + k].abs())
                                } else {
                                    None
                                }
                            }
                        })
                    })
                    .collect();
                let excess = measured.iter().filter(|&&v| v > 3.).count() as f64
                    / measured.len().max(1) as f64;
                for (gi, gate) in joint.profile.gates.iter_mut().enumerate() {
                    let first = gi * decode::NL;
                    let mut q = 0.;
                    let mut count = 0;
                    let mut bad = 0;
                    for i in first..first + decode::NL {
                        for k in 0..2 {
                            if p.problem.products[i].sigma[k] < 1e20 {
                                let v = r[2 * i + k];
                                q += v * v;
                                count += 1;
                                if v.abs() > 3. {
                                    bad += 1;
                                }
                            }
                        }
                    }
                    gate.chi_square = Some(q);
                    if count > 0 && bad as f64 / count as f64 > 0.2 {
                        gate.flags.push("large_lag_residuals".into());
                    }
                }
                let mut v = serde_json::to_value(&joint)?;
                v["shared_composition"] = serde_json::json!({"window_start_utc":time,"number_of_beams":prepared.len(),"temporal":temporal,"common_reference_job":selected[reference].id,"robust_huber_threshold":3.,"global_robust_objective":solution.robust_objective,"global_temporal_objective":solution.temporal_objective,"independent_plasma_profiles":true});
                campaign::save(&out.join("joint-fits").join(format!("{}.h5", j.id)), &v)?;
                beam_stats.push(serde_json::json!({"job":j.id,"raw_chi_square":chi,"raw_reduced_chi_square":chi/dof.max(1)as f64,"fraction_residual_components_above_3sigma":excess,"exposure_s":p.exposure}));
            }
            let summary = serde_json::json!({"schema":"syisr-ac-shared-composition-window-v1","start":time,"end":selected[0].end,"status":solution.status,"normalized_gradient":solution.normalized_gradient,"iterations":solution.iterations,"objective_history":solution.objective_history,"robust_objective":solution.robust_objective,"temporal_objective":solution.temporal_objective,"h_transition":Transition::new((*shape).clone(),[solution.shared[0],solution.shared[1]]),"helium_layer":HeliumLayer::new([solution.shared[2],solution.shared[3],solution.shared[4]]),"beams":beam_stats,"excluded_beams":skipped,"temporal":temporal,"uncertainty":"disabled","horizontal_assumption":"H+ and He+ depend on nominal altitude and elapsed window, identical across beams; Ne/Ti/Te/line-of-sight velocity independent per beam"});
            campaign::save(
                &out.join("shared-windows").join(format!("{time}.h5")),
                &summary,
            )?;
            eprintln!(
                "SHARED_COMPLETE {time} {} gradient={}",
                solution.status, solution.normalized_gradient
            );
            Ok(())
        })();
        if let Err(e) = result {
            failures += 1;
            fs::create_dir_all(out.join("failures"))?;
            fs::write(
                out.join("failures").join(format!("{time}.txt")),
                format!("{e:#}"),
            )?;
            eprintln!("SHARED_FAILED {time} {e:#}");
        }
    }
    ensure!(failures == 0, "{failures} shared windows failed");
    Ok(())
}

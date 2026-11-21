//! Conditional H-fraction checks, with every beam's plasma profiles refitted.
use anyhow::{ensure, Result};
use rustplotlib::*;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(args.len() == 3, "RESULT_DIRECTORY OUTPUT_PNG");
    let mut groups = BTreeMap::<i64, Vec<Value>>::new();
    for file in std::fs::read_dir(&args[1])? {
        let path = file?.path();
        if path.extension().is_some_and(|x| x == "h5") {
            let row: Value = syisr_ac::campaign::load(&path)?;
            ensure!(row["status"] == "converged" && row["target_achieved"] == true,
                "unresolved constraint in {}", path.display());
            groups.entry(row["time"].as_i64().unwrap()).or_default().push(row);
        }
    }
    ensure!(groups.len() == 3, "expected three diagnostic times");
    let mut fig = Figure::new(IntendedUse::ArticleWide);
    fig.width_mm = 188.; fig.height_mm = 200.; fig.font_pt = 10.;
    fig.labels.push(Label::new("SYISR: conditional H$^+$ fraction sensitivity", 94., 7.));
    fig.labels.push(Label::new("All beam Ne, Ti, Te and velocity refitted at each fraction", 94., 15.));
    for (i, (time, rows)) in groups.iter_mut().enumerate() {
        rows.sort_by(|a,b| a["target_h1400"].as_f64().unwrap().total_cmp(&b["target_h1400"].as_f64().unwrap()));
        let qmin = rows.iter().map(|x| x["spatial_penalized_objective"].as_f64().unwrap()).fold(f64::INFINITY, f64::min);
        let points: Vec<_> = rows.iter().map(|x| [x["actual_h1400"].as_f64().unwrap(), x["spatial_penalized_objective"].as_f64().unwrap()-qmin]).collect();
        let ymax = (points.iter().map(|p| p[1]).fold(0., f64::max)/50.).ceil()*50.;
        let mut axes = Axes::new([0.,1.],[0.,ymax.max(50.)], if i==2 {"Fixed H$^+$ fraction at 1400 km"} else {""}, "$\\Delta Q$");
        axes.xticks=vec![0.,0.25,0.5,0.75,1.];
        let y=31.+i as f64*48.;
        let mut panel=Panel::new(Rect::new(24.,y,152.,31.),axes);
        panel.push(Layer::Line(Line::new(points,"#0072B2")));
        fig.panels.push(panel);
        let date=chrono::DateTime::<chrono::Utc>::from_timestamp(*time,0).unwrap();
        fig.labels.push(Label::new(date.format("%d May 2024, %H:%M UTC").to_string(),100.,y-5.));
    }
    fig.labels.push(Label::new("Widths optimized; altitude priors retained; temporal priors removed",94.,181.));
    fig.labels.push(Label::new("Conditional working objective; no calibrated confidence thresholds",94.,190.));
    fig.save(Path::new(&args[2]))?;
    Ok(())
}

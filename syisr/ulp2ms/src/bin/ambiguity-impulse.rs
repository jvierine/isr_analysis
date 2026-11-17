//! Rows and columns of the same range-lag forward operator for Memo 58.
use anyhow::{ensure, Result};
use num_complex::Complex64 as C;
use rustplotlib::*;
use syisr_long_pulse::{
    ambiguity_audit::{direct_point_response, nominal_range, point_response},
    profile_response::{Ambiguity, BINS, DR, OFFSET},
};
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(
        a.len() == 3,
        "ambiguity-impulse noon-reference.h5 output-prefix"
    );
    let source = hdf5::File::open(&a[1])?;
    let get = |n: &str| source.dataset(n).unwrap().read_raw::<f64>().unwrap();
    let t = get("receiver_fir_offset_s");
    let re = get("receiver_fir_real");
    let im = get("receiver_fir_imag");
    let fir = t
        .iter()
        .enumerate()
        .map(|(j, &t)| (t, C::new(re[j], im[j])))
        .collect::<Vec<_>>();
    let filtered = Ambiguity::new(&fir)?;
    let ideal = Ambiguity::new(&[(0., C::new(1., 0.))])?;
    let indices = (150..=450).step_by(5).collect::<Vec<_>>();
    let ranges = indices
        .iter()
        .map(|&i| nominal_range(i))
        .collect::<Vec<_>>();
    let nr = ranges.len();
    let nl = 61;
    let intrinsic = [17isize, 50];
    let point_range = 600.;
    let mut ideal_columns = vec![];
    let mut columns = vec![];
    let mut maximum_error = 0f64;
    let mut peak = 0f64;
    for &k in &intrinsic {
        let mut unfiltered = vec![];
        let mut response = vec![];
        for (&i, &r) in indices.iter().zip(&ranges) {
            for l in 0..nl {
                let u = point_response(&ideal, r, l, point_range, k, true);
                let v = point_response(&filtered, r, l, point_range, k, true);
                let independent = direct_point_response(i, l, point_range, k, &fir);
                maximum_error = maximum_error.max((v - independent).norm());
                peak = peak.max(v.norm());
                unfiltered.push(u);
                response.push(v);
            }
        }
        ideal_columns.push(unfiltered);
        columns.push(response);
    }
    ensure!(
        maximum_error < 1e-12,
        "point-source replay error {maximum_error}"
    );
    let norm = ideal_columns
        .iter()
        .flatten()
        .map(|v| v.re.abs())
        .fold(0., f64::max);
    let imag_max = columns
        .iter()
        .flatten()
        .map(|v| v.im.abs() / norm)
        .fold(0., f64::max);
    let power = 10f64.powf(imag_max.max(1e-12).log10().floor());
    let imag_bound = (imag_max / power).ceil().max(1.) * power;
    let out = hdf5::File::create(format!("{}.h5", a[2]))?;
    out.new_dataset_builder()
        .with_data(&ranges)
        .create("output_slant_range_km")?;
    out.new_dataset_builder()
        .with_data(&(0..nl).map(|l| l as f64 * 10.).collect::<Vec<_>>())
        .create("output_lag_us")?;
    out.new_dataset_builder()
        .with_data(&intrinsic.map(|k| k as f64 * 10.))
        .create("input_lag_us")?;
    for (name, data) in [
        (
            "ideal_column_real_per_km",
            ideal_columns
                .iter()
                .flatten()
                .map(|v| v.re)
                .collect::<Vec<_>>(),
        ),
        (
            "filtered_column_real_per_km",
            columns.iter().flatten().map(|v| v.re).collect(),
        ),
        (
            "filtered_column_imag_per_km",
            columns.iter().flatten().map(|v| v.im).collect(),
        ),
    ] {
        out.new_dataset::<f64>()
            .shape((2, nr, nl))
            .create(name)?
            .write_raw(&data)?;
    }
    let x = (0..=nl).map(|l| -5. + 10. * l as f64).collect::<Vec<_>>();
    let dr = ranges[1] - ranges[0];
    let y = (0..=nr)
        .map(|i| ranges[0] - dr / 2. + i as f64 * dr)
        .collect::<Vec<_>>();
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 240.;
    f.height_mm = 231.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        "Forward response of an intrinsic range--lag spike",
        115.,
        5.,
    ));
    f.labels.push(Label::new(
        "Input range 600 km; pulse, FIR and trapezoidal averaging",
        115.,
        12.,
    ));
    for row in 0..2 {
        let yy = 34. + row as f64 * 90.;
        f.labels.push(Label::new(
            format!(
                "Input: one lag bin at {} microseconds; unit integrated range strength",
                intrinsic[row] * 10
            ),
            115.,
            yy - 13.,
        ));
        for col in 0..3 {
            let values = (0..nr * nl)
                .map(|j| {
                    if col == 0 {
                        ideal_columns[row][j].re / norm
                    } else if col == 1 {
                        columns[row][j].re / norm
                    } else {
                        columns[row][j].im / norm
                    }
                })
                .collect::<Vec<_>>();
            let limit = if col == 2 { imag_bound } else { 1. };
            let sc = f.scale(Scale::new(
                [-limit, limit],
                if col == 2 {
                    Normalization::Linear
                } else {
                    Normalization::Asinh { width: 0.01 }
                },
                ColorMap::diverging(),
            )?);
            let mut ax = Axes::new(
                [-5., 605.],
                [y[0], *y.last().unwrap()],
                r"Measured lag ($\mu$s)",
                if col == 0 {
                    "Output range centre (km)"
                } else {
                    ""
                },
            );
            ax.xticks = vec![0., 200., 400., 600.];
            ax.yticks = vec![400., 600., 800.];
            let xx = 21. + col as f64 * 73.;
            let mut p = Panel::new(Rect::new(xx, yy, 57., 48.), ax);
            p.push(Layer::Pcolormesh(Field::new(
                x.clone(),
                y.clone(),
                values,
                sc,
            )?));
            f.panels.push(p);
            f.labels.push(Label::new(
                [
                    "Ideal receiver: real",
                    "Inferred receiver: real",
                    "Inferred receiver: imaginary",
                ][col],
                xx + 28.5,
                yy - 6.,
            ));
            let mut cb = Colorbar::new(
                Rect::new(xx, yy + 59., 57., 2.),
                sc,
                "",
                if col == 2 {
                    vec![-limit, 0., limit]
                } else {
                    vec![-1., -0.1, 0., 0.1, 1.]
                },
            );
            cb.horizontal = true;
            f.colorbars.push(cb);
        }
    }
    f.labels.push(Label::new(
        "All panels use one amplitude unit; real colours: asinh stretch (width 0.01)",
        115.,
        215.,
    ));
    f.labels.push(Label::new(
        "Imaginary scale expanded; measured lags: 0--600 microseconds",
        115.,
        223.,
    ));
    for ext in ["pdf", "png", "svg"] {
        f.save(format!("{}-columns.{ext}", a[2]))?;
    }
    // One operator ROW: all intrinsic range/lag samples contributing to a fixed
    // measured product. Contrast it with the COLUMN impulse responses above.
    let centre = nominal_range(290);
    let measured = 17;
    let true_ranges = (0..BINS)
        .map(|s| centre + (OFFSET as f64 + s as f64 + 0.5) * DR)
        .collect::<Vec<_>>();
    let kvalues = (-40..=100).collect::<Vec<isize>>();
    let nk = kvalues.len();
    let mut row_ideal = vec![];
    let mut row_filtered = vec![];
    for &r in &true_ranges {
        for &k in &kvalues {
            row_ideal.push(point_response(&ideal, centre, measured, r, k, true));
            row_filtered.push(point_response(&filtered, centre, measured, r, k, true));
        }
    }
    let rownorm = row_ideal.iter().map(|v| v.re.abs()).fold(0., f64::max);
    let imax = row_filtered
        .iter()
        .map(|v| v.im.abs() / rownorm)
        .fold(0., f64::max);
    let power = 10f64.powf(imax.max(1e-12).log10().floor());
    let ibound = (imax / power).ceil().max(1.) * power;
    out.new_dataset_builder()
        .with_data(&true_ranges)
        .create("row_input_slant_range_km")?;
    out.new_dataset_builder()
        .with_data(&kvalues.iter().map(|&k| k as f64 * 10.).collect::<Vec<_>>())
        .create("row_input_lag_us")?;
    for (name, v) in [
        (
            "ideal_row_real_per_km",
            row_ideal.iter().map(|v| v.re).collect::<Vec<_>>(),
        ),
        (
            "filtered_row_real_per_km",
            row_filtered.iter().map(|v| v.re).collect(),
        ),
        (
            "filtered_row_imag_per_km",
            row_filtered.iter().map(|v| v.im).collect(),
        ),
    ] {
        out.new_dataset::<f64>()
            .shape((BINS, nk))
            .create(name)?
            .write_raw(&v)?;
    }
    let xx = (0..=nk).map(|k| -405. + k as f64 * 10.).collect::<Vec<_>>();
    let yy = (0..=BINS)
        .map(|s| centre + (OFFSET as f64 + s as f64) * DR)
        .collect::<Vec<_>>();
    let mut f = Figure::new(IntendedUse::ArticleWide);
    f.width_mm = 240.;
    f.height_mm = 112.;
    f.raster_dpi = 180.;
    f.labels.push(Label::new(
        format!(
            "One measured product: centre {:.2} km, lag 170 microseconds",
            centre
        ),
        115.,
        5.,
    ));
    for col in 0..3 {
        let limit = if col == 2 { ibound } else { 1. };
        let sc = f.scale(Scale::new(
            [-limit, limit],
            if col == 2 {
                Normalization::Linear
            } else {
                Normalization::Asinh { width: 0.01 }
            },
            ColorMap::diverging(),
        )?);
        let values = (0..row_ideal.len())
            .map(|j| {
                if col == 0 {
                    row_ideal[j].re / rownorm
                } else if col == 1 {
                    row_filtered[j].re / rownorm
                } else {
                    row_filtered[j].im / rownorm
                }
            })
            .collect::<Vec<_>>();
        let mut ax = Axes::new(
            [-405., 1005.],
            [yy[0], *yy.last().unwrap()],
            r"Intrinsic lag ($\mu$s)",
            if col == 0 {
                "True scattering range (km)"
            } else {
                ""
            },
        );
        ax.xticks = vec![-400., 0., 400., 800.];
        ax.yticks = vec![400., 600., 800.];
        let x = 21. + col as f64 * 73.;
        let mut p = Panel::new(Rect::new(x, 24., 57., 48.), ax);
        p.push(Layer::Pcolormesh(Field::new(
            xx.clone(),
            yy.clone(),
            values,
            sc,
        )?));
        f.panels.push(p);
        f.labels.push(Label::new(
            [
                "Ideal receiver: real",
                "Inferred receiver: real",
                "Inferred receiver: imaginary",
            ][col],
            x + 28.5,
            17.,
        ));
        let mut cb = Colorbar::new(
            Rect::new(x, 84., 57., 2.),
            sc,
            "",
            if col == 2 {
                vec![-limit, 0., limit]
            } else {
                vec![-1., -0.1, 0., 0.1, 1.]
            },
        );
        cb.horizontal = true;
        f.colorbars.push(cb);
    }
    f.labels.push(Label::new(
        "This is sensitivity in INPUT coordinates, not the output of one input spike",
        115.,
        102.,
    ));
    for ext in ["pdf", "png", "svg"] {
        f.save(format!("{}-row.{ext}", a[2]))?;
    }
    let metadata = serde_json::json!({"input_true_range_km":point_range,"input_lag_index":intrinsic,"range_spike":"Dirac with unit integrated range strength; coefficients per km; lag spike is Kronecker on 10 us grid","column_plot_normalizer_per_km":norm,"column_imaginary_plot_bound":imag_bound,"row_measured_centre_km":centre,"row_measured_lag_us":170,"row_plot_normalizer_per_km":rownorm,"row_imaginary_plot_bound":ibound,"maximum_absolute_point_replay_error_per_km":maximum_error,"maximum_filtered_point_response_per_km":peak,"plot":"real values use asinh width 0.01, one normalizer per figure; imaginary uses same unit with expanded limits; no scientific values clipped","provenance":{"reference":a[1]}});
    out.new_attr_builder()
        .with_data(serde_json::to_string(&metadata)?.as_bytes())
        .create("metadata_json")?;
    println!("{}", serde_json::to_string_pretty(&metadata)?);
    Ok(())
}

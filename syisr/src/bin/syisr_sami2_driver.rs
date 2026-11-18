//! Match native SAMI2 geometry and reference drivers to reconstructed ISR velocities.
use anyhow::{ensure, Result};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path};
use syisr_ac::campaign;
fn median(a: &[f64]) -> f64 {
    let mut v = a.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2] + v[n / 2 - 1]) / 2.
    } else {
        v[n / 2]
    }
}
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(mut a: [f64; 3]) -> [f64; 3] {
    let n = dot(a, a).sqrt();
    for v in &mut a {
        *v /= n;
    }
    a
}
fn basis(b: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let e = norm([b[1], -b[0], 0.]);
    (e, norm(cross(e, b)))
}
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(a.len() == 4, "DRIFT NATIVE_PROBE OUTPUT_ROOT");
    let mut drift: Value = campaign::load(Path::new(&a[1]))?;
    let bytes = fs::read(&a[2])?;
    let obs = drift["rows"].as_array_mut().unwrap();
    ensure!(
        bytes.len() == obs.len() * 48,
        "native10float record count mismatch"
    );
    let mut rows = vec![];
    let mut candidates: BTreeMap<i64, Vec<f64>> = BTreeMap::new();
    let mut times = BTreeMap::new();
    for (i, r) in obs.iter().enumerate() {
        let b = &bytes[i * 48..(i + 1) * 48];
        ensure!(
            u32::from_le_bytes(b[0..4].try_into()?) == 40
                && u32::from_le_bytes(b[44..48].try_into()?) == 40,
            "native record markers"
        );
        let f: Vec<f64> = b[4..44]
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
            .collect();
        let start = r["start_utc_s"].as_i64().unwrap();
        let z = r["altitude_km"].as_f64().unwrap();
        let t = (start as f64 + 900. - 1759622400.) / 3600.;
        ensure!(
            (f[0] - t).abs() < 1e-4 && (f[1] - z).abs() < 1e-3,
            "probe order differs"
        );
        let bb = syisr_ac::geomagnetic::igrf14(2025.76, 18.3492, 109.6222, z)?;
        let b_igrf = norm([bb[1], bb[0], -bb[2]]);
        let (e, m) = basis(b_igrf);
        let b_native = norm([-f[5], f[4], -f[3]]);
        let (_, mn) = basis(b_native);
        let lat = f[2].to_radians();
        let factor = lat.cos().powi(3) / (1. + 3. * lat.sin().powi(2)).sqrt();
        ensure!(factor > 0.1, "invalid native field-line mapping");
        let vel = &r["magnetic"]["velocity_m_s"];
        let vp = [vel[0].as_f64().unwrap(), vel[1].as_f64().unwrap()];
        let projected = vp[0] * dot(e, mn) + vp[1] * dot(m, mn);
        let eq = projected / factor;
        let hwm_parallel = dot([f[8], f[7], 0.], b_igrf);
        let good = r["magnetic"]["consistency_ok"] == true;
        if good && (300.0..=500.).contains(&z) {
            candidates.entry(start).or_default().push(eq);
        }
        times.insert(start, f[6]);
        rows.push(json!({"start_utc_s":start,"altitude_km":z,"native_magnetic_latitude_deg":f[2],"native_field_mapping":factor,"native_meridional_enu":mn,"igrf_meridional_projection":dot(m,mn),"model_equatorial_longitude_deg":f[9],"fejer_scherliess_equatorial_m_s":f[6],"fejer_scherliess_local_meridional_m_s":f[6]*factor*dot(m,mn),"measured_equatorial_m_s":eq,"measured_magnetic_m_s":vel,"measured_geographic_m_s":r["geographic"]["velocity_m_s"],"hwm_north_m_s":f[7],"hwm_east_m_s":f[8],"hwm_along_igrf_b_m_s":hwm_parallel,"accepted":good}));
    }
    let mut driver = vec![];
    for (start, baseline) in times {
        let vals = candidates.get(&start).map(|v| v.as_slice()).unwrap_or(&[]);
        let good = vals.len() >= 10;
        let measured = if good { median(vals) } else { baseline };
        let mad = if good {
            Some(median(
                &vals
                    .iter()
                    .map(|v| (v - measured).abs())
                    .collect::<Vec<_>>(),
            ))
        } else {
            None
        };
        driver.push(json!({"start_utc_s":start,"model_hour_center":(start as f64+900.-1759622400.)/3600.,"equatorial_measured_m_s":measured,"fejer_equatorial_m_s":baseline,"accepted_altitude_samples":vals.len(),"altitude_mad_m_s":mad,"measurement_used":good}));
    }
    let out = Path::new(&a[3]);
    fs::create_dir_all(out)?;
    let n = driver.len();
    for (sign, name) in [(1., "measured"), (-1., "measured-reversed")] {
        let mut table = format!("{}\n", n + 2);
        let first = &driver[0];
        table += &format!(
            "{:.9} {:.9} {}\n",
            first["model_hour_center"].as_f64().unwrap() - 0.25,
            sign * first["equatorial_measured_m_s"].as_f64().unwrap(),
            u8::from(first["measurement_used"] == true)
        );
        for row in &driver {
            table += &format!(
                "{:.9} {:.9} {}\n",
                row["model_hour_center"].as_f64().unwrap(),
                sign * row["equatorial_measured_m_s"].as_f64().unwrap(),
                u8::from(row["measurement_used"] == true)
            );
        }
        let last = &driver[n - 1];
        table += &format!(
            "{:.9} {:.9} {}\n",
            last["model_hour_center"].as_f64().unwrap() + 0.25,
            sign * last["equatorial_measured_m_s"].as_f64().unwrap(),
            u8::from(last["measurement_used"] == true)
        );
        fs::create_dir_all(out.join(name))?;
        fs::write(out.join(name).join("measured-drift.inp"), table)?;
    }
    let used = driver
        .iter()
        .filter(|v| v["measurement_used"] == true)
        .count();
    let result = json!({"schema":"syisr-sami2-measured-driver-v1","rows":rows,"driver":driver,"metadata":{"native_model":"SAMI2-1.00 eccentric dipole and its local cos^3/sqrt mapping, nativeequatorial longitude and Fejer-Scherliess routine; HWM14 atSanya","forcing":"Median300--500km qualified perpendicular ionvelocity projected fromIGRF14 magnetic east/meridional basis onto nativeSAMI2meridional direction, dividedby nativefield mapping. Along-field ionmotion excluded. Minimum10accepted height samples perwindow; theyare correlated,not independent uncertainty observations.","time":"UT modelhours from2025-10-05T00:00UTC. Measuredwindow centers linearlyinterpolated, firstandlastquarterhour heldinside observedwindowedges. Outsideandunqualifiedbrackets retainFejer. Measuredforcingnot repeatedduringspinup.","sign":"CurrentACsoftwareconvention and explicitnegative-sign sensitivity. Absolutehardwarepolarityunverified.","controls":"HWM14andallsolar,geomagnetic,chemistryandgridinputs heldfixed; onlyscalarmeridionalperpendicularforcingchanges. Zonaladvectionnotimplementedbythis2Dmodel.","comparison":"HWMneutralwindisnotionvelocity. Along-Bprojectionshownasreferenceonlywithoutmomentum-equationwindinference.","daily_inputs":{"F107":133.,"F107a":154.3,"Ap":4.,"day_of_year":279,"year":2025}},"measured_windows_used":used});
    campaign::save(&out.join("drivers.h5"), &result)?;
    println!("measured forcing {used}/{n} windows");
    Ok(())
}

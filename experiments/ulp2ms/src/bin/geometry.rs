use anyhow::{ensure, Result};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    ensure!(
        a.len() == 3 || a.len() == 6,
        "geometry lpm.h5 geometry.h5 [minimum_range_km maximum_range_km step_km]"
    );
    let h = hdf5::File::open(&a[1])?;
    let r = h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let t = h.dataset("time_utc_s")?.read_raw::<f64>()?;
    let grid = if a.len() == 6 {
        [a[3].parse()?, a[4].parse()?, a[5].parse()?]
    } else {
        [350., 1450., 50.]
    };
    syisr_long_pulse::geometry::write(
        std::path::Path::new(&a[2]),
        &r,
        (t[0] + t[t.len() - 1]) / 2.,
        grid[0],
        grid[1],
        grid[2],
    )
}

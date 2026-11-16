use anyhow::Result;
fn main() -> Result<()> {
    for el in [30., 45., 60., 75., 90.] {
        let grid = syisr_ac::grid::build(el, 1000.);
        let n = grid.iter().filter(|g| g.altitude_km < 400.).count();
        print!(
            "elevation{el:.0}: {} native bottom gates,{} total; widths",
            n,
            grid.len()
        );
        for h in [300., 500., 750., 975.] {
            let g = grid
                .iter()
                .min_by(|a, b| {
                    (a.altitude_km - h)
                        .abs()
                        .total_cmp(&(b.altitude_km - h).abs())
                })
                .unwrap();
            print!(
                " {:.0}km:{:.2}km",
                g.altitude_km,
                g.upper_altitude_km - g.lower_altitude_km
            );
        }
        println!();
    }
    Ok(())
}

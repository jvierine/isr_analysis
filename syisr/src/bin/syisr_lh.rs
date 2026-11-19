use anyhow::{ensure, Result};
use std::path::Path;
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    if a.len() == 3 && a[1] == "check" {
        return syisr_ac::lower_hybrid::validation_table(Path::new(&a[2]));
    }
    ensure!(
        a.len() == 5 || a.len() == 6,
        "ROOT UTC_WINDOW ALTITUDE_KM OUTPUT_H5 [EFFECTIVE_BGK_RATE_S]"
    );
    syisr_ac::lower_hybrid::fit(
        Path::new(&a[1]),
        a[2].parse()?,
        a[3].parse()?,
        Path::new(&a[4]),
        if a.len() == 6 { a[5].parse()? } else { 1000. },
    )
}

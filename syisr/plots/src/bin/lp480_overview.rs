//! Contact sheet for inspecting scientific exports; full-resolution maps remain separate.
use anyhow::{Result, ensure};
fn main() -> Result<()> {
    let a = std::env::args().collect::<Vec<_>>();
    ensure!(a.len() == 3, "PLOT_DIRECTORY OUTPUT_PNG");
    let mut files = std::fs::read_dir(&a[1])?
        .map(|p| p.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.retain(|p| {
        p.extension().is_some_and(|e| e == "png")
            && p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("beam-")
    });
    files.sort();
    ensure!(!files.is_empty(), "no beam figures");
    let (width, height, gap) = (750u32, 1070u32, 12u32);
    let rows = (files.len() as u32).div_ceil(3);
    let mut sheet = image::RgbaImage::from_pixel(
        3 * (width + gap) + gap,
        rows * (height + gap) + gap,
        image::Rgba([255, 255, 255, 255]),
    );
    for (i, p) in files.iter().enumerate() {
        let source = image::open(p)?.to_rgba8();
        let tile = image::imageops::resize(
            &source,
            width,
            height,
            image::imageops::FilterType::Lanczos3,
        );
        image::imageops::overlay(
            &mut sheet,
            &tile,
            (gap + (i as u32 % 3) * (width + gap)) as i64,
            (gap + (i as u32 / 3) * (height + gap)) as i64,
        );
    }
    sheet.save(&a[2])?;
    Ok(())
}

// Display resampling only. Never modifies the scientific field or native PNG.
use anyhow::Result;
use image::imageops::FilterType;
fn main()->Result<()>{let a:Vec<String>=std::env::args().collect();let img=image::open(&a[1])?;let width=1800;let height=(img.height()as f64*width as f64/img.width()as f64).round()as u32;img.resize_exact(width,height,FilterType::Lanczos3).save(&a[2])?;println!("PREVIEW {} x {} from {} x {}",width,height,img.width(),img.height());Ok(())}

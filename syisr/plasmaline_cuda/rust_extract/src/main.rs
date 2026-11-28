// Scalar visualization transport only: preserve native range/frequency cells.
// Raw and decoded scientific power remain in their original HDF5 files.
use anyhow::{Result, ensure};
use hdf5::types::VarLenUnicode;
use std::path::PathBuf;
fn main()->Result<()> {
 let a:Vec<String>=std::env::args().collect();let root=PathBuf::from(&a[1]);let dest=PathBuf::from(&a[2]);std::fs::create_dir_all(&dest)?;
 let settings=hdf5::File::open(root.join("display_settings_frequency_normalized.h5"))?;
 let limits=settings.dataset("color_limits_db")?.read_raw::<f64>()?;
 let background=settings.dataset("decoded_frequency_background_power_ratio")?.read_raw::<f64>()?;
 let sel=hdf5::File::open(root.join("selection.h5"))?;let paths=sel.dataset("matched_path")?.read_raw::<VarLenUnicode>()?;
 let raw_display=a.get(3).is_some_and(|s|s=="raw");let only:Option<usize>=a.get(4).map(|s|s.parse()).transpose()?;
 for (i,path) in paths.iter().enumerate(){if only.is_some_and(|j|i+1!=j){continue;}
  let f=hdf5::File::open(path.as_str())?;let freq=f.dataset("frequency_offset_hz")?.read_raw::<f64>()?;let range=f.dataset("slant_range_km")?.read_raw::<f64>()?;
  let nf=freq.len();let nr=range.iter().take_while(|&&r|r<=800.).count();let p=f.dataset("whitened_matched_power_ratio")?.read_raw::<f64>()?;
  ensure!(nf==8192&&nr>0,"unsupported display grid");let mut display=Vec::with_capacity(nr*nf);
  for r in 0..nr{for k in 0..nf{let x=p[r*nf+k]/if raw_display{1.}else{background[i*nf+k]};display.push(if x.is_finite()&&x>0.{((10.*x.log10()-limits[0])/(limits[1]-limits[0])*254.).round().clamp(0.,254.)as u8}else{255});}}
  let out=hdf5::File::create(dest.join(format!("profile-{:02}-native.h5",i+1)))?;
  out.new_dataset::<u8>().shape((nr,nf)).chunk((16,nf)).deflate(1).create("display_index")?.write_raw(&display)?;
  for (name,values) in [("frequency_hz",freq.as_slice()),("slant_range_km",&range[..nr]),("color_limits_db",limits.as_slice())]{out.new_dataset::<f64>().shape(values.len()).create(name)?.write_raw(values)?;}
  for name in ["integration_start_utc_s","accepted_pulses"]{let v=f.attr(name)?.read_scalar::<f64>()?;out.new_attr::<f64>().create(name)?.write_scalar(&v)?;}
  out.new_attr::<f64>().create("range_step_us")?.write_scalar(&2.)?;
  println!("EXTRACTED profile {}: {} x {}, <=800km; native cells, 255 display color levels",i+1,nf,nr);
 }
 Ok(())
}

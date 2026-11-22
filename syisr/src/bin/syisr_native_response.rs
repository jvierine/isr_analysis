//! Native 1 us receiver FIR inside the finite-range 32-baud AC ambiguity.
use anyhow::Result;
use syisr_ac::{decode,native_receiver_response};
fn main()->Result<()> {
 let a:Vec<_>=std::env::args().collect();let codes=decode::phase(std::path::Path::new(&a[1]))?;
 let file=hdf5::File::open(&a[2])?;let h=file.dataset("voltage_impulse_response")?.read_raw::<f64>()?;
 let dc=h.iter().sum::<f64>();let taps:Vec<_>=h.iter().enumerate().map(|(i,v)|(i as i32-131,*v/dc)).collect();
 let k=native_receiver_response::build(&codes,&taps,62,-48,48)?;
 let nr=(k.last_cell-k.first_cell+1) as usize;let nu=(k.last_intrinsic_us-k.first_intrinsic_us+1) as usize;
 let out=hdf5::File::create(&a[3])?;
 out.new_dataset::<f64>().shape((62,nr,nu)).deflate(4).create("coefficient")?.write_raw(&k.coefficients)?;
 out.new_dataset_builder().with_data(&(k.first_cell..=k.last_cell).map(|i|i as f64*20e-6).collect::<Vec<_>>()).create("source_delay_offset_s")?;
 out.new_dataset_builder().with_data(&(k.first_intrinsic_us..=k.last_intrinsic_us).map(|i|i as f64*1e-6).collect::<Vec<_>>()).create("intrinsic_lag_s")?;
 out.new_dataset_builder().with_data(&(1..=62).map(|i|i as f64*1e-5).collect::<Vec<_>>()).create("measured_lag_s")?;
 out.new_attr::<f64>().create("tap_origin_removed_us")?.write_scalar(&131.)?;
 println!("{}x{}x{} coefficients",62,nr,nu);Ok(())
}

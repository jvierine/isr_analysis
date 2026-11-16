//! Independent timing diagnostic; input paths and generator remain metadata only.
use anyhow::{ensure,Result};
use hdf5::File;
use rustplotlib::*;
use std::path::Path;
fn main()->Result<()>{
    let a:Vec<_>=std::env::args().collect();ensure!(a.len()==3,"plot-pulse-edge AUDIT.h5 OUTPUT-STEM");
    let h=File::open(&a[1])?;let g=h.group("pulse_edge")?;
    let mut f=Figure::new(IntendedUse::Custom{width_mm:210.,font_pt:10.});
    f.height_mm=130.;f.raster_dpi=180.;f.tight_export=false;
    f.labels.push(Label::new("Coherent near-gate pulse edge: independent timing diagnostic",105.,8.));
    let mut p=Panel::new(Rect::new(27.,25.,165.,73.),Axes::new([0.,300.],[0.,1.25],"Sample time minus commanded pulse end (microseconds)","Power / initial plateau"));
    for (name,time,color) in [("control_normalized_power","control_time_after_commanded_pulse_end_us","#bbbbbb"),("new_normalized_power","new_time_after_commanded_pulse_end_us","#0072B2")] {
        let d=g.dataset(name)?;let shape=d.shape();let data=d.read_raw::<f64>()?;let times=g.dataset(time)?.read_raw::<f64>()?;
        for row in 0..shape[0] {
            let x=if times.len()==shape[1]{&times[..]}else{&times[row*shape[1]..(row+1)*shape[1]]};
            let mut line=Line::new(x.iter().zip(&data[row*shape[1]..(row+1)*shape[1]]).map(|(x,y)|[*x,*y]).collect(),color);
            line.width_pt=0.8;p.push(Layer::Line(line));
        }
    }
    let mut edge=Line::new(vec![[160.,0.],[160.,1.25]],"#333333");edge.dash=Some("2 2".into());p.push(Layer::Line(edge));
    let mut legend=Legend::new(vec![LegendEntry::line("480 us pulse: five beams","#0072B2"),LegendEntry::line("2000 us pulse: 12 controls","#bbbbbb")]);
    legend.position=LegendPosition::AboveCenter;legend.columns=2;p.legend=Some(legend);f.panels.push(p);
    f.labels.push(Label::new("Dashed: provisional 160 us delay (24.0 km monostatic range)",105.,115.));
    f.labels.push(Label::new("Assumes zero-range transmitter leakage; hardware timing reference unverified",105.,123.));
    let out=Path::new(&a[2]);f.save(out.with_extension("png"))?;f.save(out.with_extension("svg"))?;Ok(())
}

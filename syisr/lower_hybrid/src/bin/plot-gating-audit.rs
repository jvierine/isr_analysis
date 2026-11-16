//! Separate an observed common offset from numerical gate-centroid tests.
use anyhow::{ensure,Result};
use hdf5::File;
use rustplotlib::*;
use std::path::Path;

fn main()->Result<()>{
    let a:Vec<_>=std::env::args().collect();ensure!(a.len()==3,"plot-gating-audit AUDIT.h5 OUTPUT-STEM");
    let h=File::open(&a[1])?;let out=Path::new(&a[2]);
    let range=h.dataset("slant_range_km")?.read_raw::<f64>()?;
    let rows=h.dataset("narrowing_summary")?.read_raw::<f64>()?;
    let geometry=h.dataset("beam_geometry")?.read_raw::<f64>()?;
    let profile=h.dataset("central_fraction_profile")?.read_raw::<f64>()?;
    let point=h.dataset("injected_point_power_response")?.read_raw::<f64>()?;
    let nr=range.len();let median=h.attr("median_empirical_offset_km")?.read_scalar::<f64>()?;
    let mut f=Figure::new(IntendedUse::Custom{width_mm:230.,font_pt:10.});
    f.height_mm=193.;f.raster_dpi=190.;f.tight_export=false;
    f.labels.push(Label::new("SYISR range-gating audit",115.,7.));
    f.labels.push(Label::new("Observed narrowing and synthetic point-response check",115.,15.));
    let mut top=Panel::new(Rect::new(25.,36.,125.,53.),Axes::new([-100.,100.],[0.25,0.65],"Range minus nominal perpendicular range (km)","Central spectral fraction"));
    let mut lower=Panel::new(Rect::new(25.,118.,125.,47.),Axes::new([-100.,100.],[0.,1.06],"Gate range minus injected true range (km)","Normalized response"));
    let colors=["#0072B2","#D55E00","#009E73","#CC79A7","#6a512b"];
    let mut entries=Vec::new();
    for b in 0..5 {
        let target=rows[b*4];let altitude=geometry[b*7] as usize;
        let pts:Vec<_>=range.iter().enumerate().filter(|(_,r)|**r>230. && (**r-target).abs()<=110.)
            .map(|(j,r)|[*r-target,profile[b*nr+j]]).collect();
        let mut line=Line::new(pts,colors[b]);line.width_pt=1.25;top.push(Layer::Line(line));
        let pts:Vec<_>=range.iter().enumerate().filter(|(_,r)|(**r-target).abs()<=120.)
            .map(|(j,r)|[*r-target,point[b*nr+j]]).collect();
        let mut line=Line::new(pts,colors[b]);line.width_pt=1.1;lower.push(Layer::Line(line));
        entries.push(LegendEntry::line(format!("{altitude} km beam"),colors[b]));
    }
    for p in [&mut top,&mut lower] {
        let limits=if p.rect.y<100.{[0.25,0.65]}else{[0.,1.06]};
        let mut zero=Line::new(vec![[0.,limits[0]],[0.,limits[1]]],"#555555");zero.width_pt=0.9;zero.dash=Some("2 1.5".into());p.push(Layer::Line(zero));
    }
    let mut offset=Line::new(vec![[median,0.25],[median,0.65]],"#111111");
    offset.width_pt=0.8;offset.dash=Some("0.5 1.5".into());top.push(Layer::Line(offset));
    let mut legend=Legend::new(entries);legend.columns=3;legend.position=LegendPosition::AboveCenter;top.legend=Some(legend);
    f.panels.push(top);f.panels.push(lower);
    f.labels.push(Label::new("Measured local offsets",189.,34.));
    for b in 0..5 {
        f.labels.push(Label::new(format!("{} km: +{:.1} km",geometry[b*7] as usize,rows[b*4+2]),189.,43.+b as f64*8.));
    }
    f.labels.push(Label::new(format!("Median +{median:.1} km"),189.,88.));
    f.labels.push(Label::new("Equivalent delay",189.,104.));
    f.labels.push(Label::new(format!("{:.0} microseconds",h.attr("median_equivalent_delay_us")?.read_scalar::<f64>()?),189.,112.));
    f.labels.push(Label::new("Candidate, uncalibrated",189.,124.));
    f.labels.push(Label::new("Gate spacing 18.0 km",189.,143.));
    f.labels.push(Label::new("480 microsecond pulse",189.,151.));
    f.labels.push(Label::new("Dashed: nominal zero; dotted: empirical median offset",115.,187.));
    f.save(out.with_extension("png"))?;f.save(out.with_extension("svg"))?;
    Ok(())
}

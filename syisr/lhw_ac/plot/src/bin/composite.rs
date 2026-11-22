//! Stitched altitude-sector movie. Scientific provenance stays in HDF5.
use anyhow::{ensure,Result};
use chrono::{DateTime,Utc};
use hdf5::File;
use rustplotlib::*;
use std::{env,path::Path};
fn clock(t:f64)->String {DateTime::<Utc>::from_timestamp(t as i64,0).unwrap().format("%H:%M:%S").to_string()}
fn main()->Result<()> {
    let args:Vec<_>=env::args().collect();
    ensure!(args.len()==3,"composite INPUT.h5 OUTPUT.mp4");
    let h=File::open(&args[1])?;
    let z=h.dataset("altitude_km")?.read_raw::<f64>()?;
    let freq=h.dataset("frequency_hz")?.read_raw::<f64>()?;
    let starts=h.dataset("window_start_utc_s")?.read_raw::<f64>()?;
    let ends=h.dataset("window_end_utc_s")?.read_raw::<f64>()?;
    let data=h.dataset("normalized_psd_1e4_per_hz")?.read_raw::<f64>()?;
    let cuts=h.dataset("cut_normalized_psd_1e4_per_hz")?.read_raw::<f64>()?;
    let heights=h.dataset("cut_altitude_km")?.read_raw::<f64>()?;
    let linear=h.dataset("cut_linear_limits")?.read_raw::<f64>()?;
    let log=h.dataset("cut_log_limits")?.read_raw::<f64>()?;
    let (nz,nf,nc)=(z.len(),freq.len(),heights.len());
    let individual=h.attr("beam_nominal_target_altitude_km").ok().map(|a|a.read_scalar::<i64>()).transpose()?;
    let feature_cuts=h.attr("observed_feature_cuts").ok().map(|a|a.read_scalar::<i64>()).transpose()?.unwrap_or(0)!=0;
    let pseudo=h.attr("pseudo_perpendicular").ok().map(|a|a.read_scalar::<i64>()).transpose()?.unwrap_or(0)!=0;
    let heading=if let Some(target)=individual {format!("{target} km beam | az {:.3}, el {:.3}",h.attr("recorded_azimuth_deg")?.read_scalar::<f64>()?,h.attr("physical_elevation_deg").or_else(|_|h.attr("recorded_elevation_deg"))?.read_scalar::<f64>()?)}else{if pseudo{"pseudo perpendicular-to-B"}else{"stitched sectors"}.into()};
    let colors:Vec<&str>=if individual.is_some(){vec!["#984ea3","#1f78b4","#e66101"]}else{vec!["#1f78b4","#33a02c","#e66101","#984ea3","#e7298a","#a6761d"]};
    ensure!(nc==colors.len() && data.len()==starts.len()*nz*nf,"composite shape mismatch");
    let df=(freq[1]-freq[0])/1000.;
    let ze:Vec<_>=(0..=nz).map(|i|if i==0{z[0]-(z[1]-z[0])/2.}else if i==nz{z[nz-1]+(z[nz-1]-z[nz-2])/2.}else{(z[i-1]+z[i])/2.}).collect();
    let fe:Vec<_>=(0..=nf).map(|i|freq[0]/1000.-df/2.+i as f64*df).collect();
    let mut writer=AnimationWriter::new(Path::new(&args[2]),AnimationOptions{input_fps:8.,output_fps:24.,..Default::default()})?;
    for frame in 0..starts.len() {
        let mut fig=Figure::new(IntendedUse::Custom{width_mm:304.8,font_pt:10.});
        fig.height_mm=171.45;fig.raster_dpi=160.;fig.tight_export=false;
        let a=DateTime::<Utc>::from_timestamp(starts[frame] as i64,0).unwrap();
        let b=DateTime::<Utc>::from_timestamp(ends[frame] as i64,0).unwrap();
        let date=if a.date_naive()==b.date_naive(){a.format("%d %b %Y").to_string()}else{format!("{}--{}",a.format("%d %b %Y"),b.format("%d %b %Y"))};
        fig.labels.push(Label::new(format!("SYISR AC | {heading} | {date} {}--{} UTC | {:0.0} min",clock(starts[frame]),clock(ends[frame]),(ends[frame]-starts[frame])/60.),152.4,6.));
        for (i,&height) in heights.iter().enumerate() {
            let (text,x,y)=if individual.is_some(){match i{0=>(format!("{height:.0} km"),42.,13.),1=>(if feature_cuts{format!("Feature cut: {height:.0} km")}else{format!("Predicted perpendicular: {height:.1} km")},96.,13.),_=>(format!("Reference: {height:.1} km (+900 us)"),78.,19.)}}else{(format!("{height:.0} km"),42.+(i%3) as f64*36.,13.+(i/3) as f64*6.)};
            let mut label=Label::new(text,x,y);
            label.color=colors[i].into();fig.labels.push(label);
        }
        for logarithmic in [true,false] {
            let limits=if logarithmic{[log[0],log[1]]}else{[linear[frame*2],linear[frame*2+1]]};
            let mut axes=Axes::new([fe[0],fe[nf]],limits,if logarithmic{""}else{"Doppler frequency (kHz)"},"");
            axes.xticks=vec![-40.,-20.,0.,20.,40.];
            if logarithmic {axes.set_yscale(AxisScale::Log10)?;}
            let mut panel=Panel::new(if logarithmic{Rect::new(24.,31.,108.,49.)}else{Rect::new(24.,103.,108.,55.)},axes);
            for (i,color) in colors.iter().enumerate() {
                let points=(0..nf).map(|j|{let v=cuts[(frame*nc+i)*nf+j];[freq[j]/1000.,if logarithmic && v<=0.{f64::NAN}else{v}]}).collect();
                let mut line=Line::new(points,*color);line.width_pt=1.;panel.push(Layer::Line(line));
            }
            if !logarithmic {let mut zero=Line::new(vec![[fe[0],0.],[fe[nf],0.]],"#9b9b9b");zero.width_pt=0.5;zero.opacity=0.5;panel.push(Layer::Line(zero));}
            fig.panels.push(panel);
            fig.labels.push(Label::new(if logarithmic{"Logarithmic (positive estimates)"}else{"Linear"},78.,if logarithmic{26.}else{97.}));
            let mut label=Label::new(r"Normalized PSD ($10^{-4}$ Hz$^{-1}$)",4.,if logarithmic{55.5}else{130.5});label.rotation=-90.;fig.labels.push(label);
        }
        let scale=fig.scale(Scale::new([-0.01,0.05],Normalization::Asinh{width:0.001},ColorMap::inferno())?);
        let mut axes=Axes::new([fe[0],fe[nf]],[80.,z[nz-1]],"Doppler frequency (kHz)","Altitude (km)");
        axes.xticks=vec![-40.,-20.,0.,20.,40.];axes.yticks=vec![80.,100.,150.,200.,250.,300.,350.,400.,450.,500.].into_iter().filter(|v|*v<=z[nz-1]).collect();
        let mut panel=Panel::new(Rect::new(174.,19.,108.,139.),axes);
        panel.push(Layer::Pcolormesh(Field::new(fe.clone(),ze.clone(),data[frame*nz*nf..(frame+1)*nz*nf].to_vec(),scale)?));
        for (i,&height) in heights.iter().enumerate() {
            let mut line=Line::new(vec![[fe[0],height],[fe[nf],height]],colors[i]);line.width_pt=0.6;line.opacity=0.45;line.dash=Some("2.0 1.5".into());panel.push(Layer::Line(line));
        }
        fig.panels.push(panel);
        fig.labels.push(Label::new(if individual.is_some(){"Individual beam"}else{"Near-perpendicular sectors"},228.,13.));
        fig.colorbars.push(Colorbar::new(Rect::new(285.,19.,3.,139.),scale,r"Normalized PSD ($10^{-4}$ Hz$^{-1}$)",vec![-0.01,0.,0.001,0.01,0.05]));
        writer.push(&fig)?;
        println!("Composite frame {frame}");
    }
    writer.finish()?;
    Ok(())
}

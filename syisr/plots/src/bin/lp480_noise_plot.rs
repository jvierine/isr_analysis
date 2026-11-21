//! Receiver-noise diagnostics; these curves are not an adopted signal filter.
use anyhow::{ensure, Result};
use rustplotlib::*;
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};
fn main() -> Result<()> {
    let args:Vec<_>=std::env::args().collect();
    ensure!(args.len()==3,"NOISE_AUDIT_H5 OUTPUT_PNG");
    let data:Value=syisr_ac::campaign::load(Path::new(&args[1]))?;
    let reference=data["quiet_zero_lag_adu2"].as_f64().unwrap();
    let mut days=BTreeMap::<i64,Vec<[f64;2]>>::new();
    for row in data["rows"].as_array().unwrap(){let t=row[0].as_i64().unwrap();
        days.entry(t.div_euclid(86400)).or_default().push([(t.rem_euclid(86400)) as f64/3600.,row[2][0][0].as_f64().unwrap()/reference]);}
    let colors=["#0072B2","#D55E00","#009E73","#CC79A7"];
    let ymax=10f64.powf(days.values().flatten().map(|p|p[1]).fold(10.,f64::max).log10().ceil());
    let mut figure=Figure::new(IntendedUse::ArticleWide);
    figure.width_mm=195.;figure.height_mm=204.;figure.font_pt=10.;
    figure.labels.push(Label::new("SYISR zenith: solar-transit noise diagnostics",98.,6.));
    figure.labels.push(Label::new("11--14 May 2024",98.,13.));
    let mut axes=Axes::new([3.5,6.],[1.,ymax],"UTC","Noise power / quiet reference");
    axes.set_yscale(AxisScale::Log10)?;axes.yticks=vec![1.,10.,100.,1000.].into_iter().filter(|v|*v<=ymax).collect();
    axes.xticks=vec![3.5,4.,4.5,5.,5.5,6.];axes.xticklabels=Some(vec!["03:30","04:00","04:30","05:00","05:30","06:00"].into_iter().map(String::from).collect());
    let mut panel=Panel::new(Rect::new(25.,24.,143.,34.),axes);
    for (i,(_,points)) in days.iter().enumerate(){let mut segment=Vec::new();
        for p in points {if segment.last().is_some_and(|last:&[f64;2]|p[0]-last[0]>15./3600.){
            if segment.len()>1{panel.push(Layer::Line(Line::new(segment.clone(),colors[i%4])));}segment.clear();}
            segment.push(*p);}
        if segment.len()>1{panel.push(Layer::Line(Line::new(segment,colors[i%4])));}}
    let solar_hour=4.+37./60.+50./3600.;
    panel.push(Layer::Line(Line::new(vec![[solar_hour,1.],[solar_hour,ymax]],"black")));
    figure.labels.push(Label::new("Sun nearest zenith",25.+143.*(solar_hour-3.5)/2.5,20.));
    figure.panels.push(panel);
    for component in 0..2{
        let refined=data["window_detrended_noise_response"]["per_day"].is_object();
        let mut extent=0.01f64;
        for(day,_)in &days{let slope=if refined{&data["window_detrended_noise_response"]["per_day"][day.to_string()]["slope"]}else{&data["per_day_within_dwell_slopes"][day.to_string()]};
            for l in 1..=15{extent=extent.max(slope[l][component].as_f64().unwrap().abs()).max(data["quiet_normalized_acf"][l][component].as_f64().unwrap().abs());}}
        let bound=(extent*100.).ceil()/100.;
        let mut axes=Axes::new([10.,150.],[-bound,bound],r"Lag ($\mu$s)",if component==0{"Real noise-ACF slope"}else{"Imaginary noise-ACF slope"});
        axes.xticks=vec![10.,50.,100.,150.];axes.yticks=vec![-bound,0.,bound];
        let mut panel=Panel::new(Rect::new(25.,82.+component as f64*50.,143.,31.),axes);
        let lags=data["lag_us"].as_array().unwrap();
        let points=lags.iter().enumerate().filter(|(_,l)|l.as_f64().unwrap()>=10.&&l.as_f64().unwrap()<=150.)
            .map(|(l,t)|[t.as_f64().unwrap(),data["quiet_normalized_acf"][l][component].as_f64().unwrap()]).collect();
        panel.push(Layer::Line(Line::new(points,"#777777")));
        for(i,(day,_))in days.iter().enumerate(){let slope=if refined{&data["window_detrended_noise_response"]["per_day"][day.to_string()]["slope"]}else{&data["per_day_within_dwell_slopes"][day.to_string()]};
            let points=lags.iter().enumerate().filter(|(_,l)|l.as_f64().unwrap()>=10.&&l.as_f64().unwrap()<=150.)
                .map(|(l,t)|[t.as_f64().unwrap(),slope[l][component].as_f64().unwrap()]).collect();
            panel.push(Layer::Line(Line::new(points,colors[i%4])));}
        figure.panels.push(panel);}
    for(i,(day,_))in days.iter().enumerate(){let date=chrono::DateTime::<chrono::Utc>::from_timestamp(day*86400,0).unwrap();
        let mut label=Label::new(date.format("%d May").to_string(),43.+i as f64*35.,180.);label.color=colors[i%4].into();figure.labels.push(label);}
    figure.labels.push(Label::new("Gray: quiet ACF / R(0); colors: detrended noise-response slopes",98.,189.));
    figure.labels.push(Label::new("Diagnostic only; receiver stationarity and solar spectrum remain to be checked",98.,197.));
    figure.save(&args[2])?;Ok(())
}

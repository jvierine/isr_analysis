//! Exact pointwise-H constraints, with nuisance plasma parameters refitted.
use crate::{campaign, coupled, h_transition::{Shape,CENTER_MIN,CENTER_MAX,WIDTH_MIN,WIDTH_MAX}, lp480::Decoded,
    lp480_fit::{Background,Reference,WindowFit,prepare,project_initial}, sharedcomposition};
use anyhow::{ensure,Result};
use serde::{Deserialize,Serialize};
use std::{collections::BTreeMap,path::{Path,PathBuf},sync::Arc};

fn numerator(shape:&Shape,z:f64,c:f64)->f64{(shape.coordinate(z)-shape.coordinate(c))*shape.height(c).0}
// Split into monotone pieces. Strong scale-height skew can give two possible
// transition heights at the same width and pointwise fraction; retain both.
fn center_cuts(shape:&Shape,z:f64)->Vec<f64>{
    let mut cuts=vec![CENTER_MIN,CENTER_MAX];
    cuts.extend(shape.altitude_km.iter().copied().filter(|c|*c>CENTER_MIN&&*c<CENTER_MAX));
    cuts.sort_by(f64::total_cmp);cuts.dedup();let segments=cuts.clone();
    for pair in segments.windows(2){let(a,b)=(pair[0],pair[1]);let slope=shape.height((a+b)*0.5).1;
        if slope<=0.{continue;}let target=shape.coordinate(z)-1./slope;
        if target>shape.coordinate(a)&&target<shape.coordinate(b){let(mut lo,mut hi)=(a,b);
            for _ in 0..60{let m=(lo+hi)*0.5;if shape.coordinate(m)<target{lo=m;}else{hi=m;}}
            cuts.push((lo+hi)*0.5);}}
    cuts.sort_by(f64::total_cmp);cuts.dedup_by(|a,b|(*a-*b).abs()<1e-9);cuts
}
pub fn transition_roots(shape:&Shape,z:f64,fraction:f64,width:f64)->Vec<[f64;2]>{
    let target=width*(fraction/(1.-fraction)).ln();let cuts=center_cuts(shape,z);let mut roots=Vec::new();
    for pair in cuts.windows(2){let(mut lo,mut hi)=(pair[0],pair[1]);let fl=numerator(shape,z,lo)-target;let fh=numerator(shape,z,hi)-target;
        if fl.abs()<1e-9{roots.push(Shape::encode(lo,width));}
        if fl*fh<0.{for _ in 0..60{let m=(lo+hi)*0.5;let fm=numerator(shape,z,m)-target;
            if fl*fm>0.{lo=m;}else{hi=m;}}
            roots.push(Shape::encode((lo+hi)*0.5,width));}
        if fh.abs()<1e-9{roots.push(Shape::encode(hi,width));}}
    roots.sort_by(|a,b|a[0].total_cmp(&b[0]));roots.dedup_by(|a,b|(a[0]-b[0]).abs()<1e-8);roots
}
fn maximum_width(shape:&Shape,z:f64,fraction:f64)->f64{
    if (fraction-0.5).abs()<1e-12{return WIDTH_MAX;}
    let logit=(fraction/(1.-fraction)).ln();
    center_cuts(shape,z).iter().map(|c|numerator(shape,z,*c)/logit).fold(0.,f64::max).min(WIDTH_MAX)
}
#[derive(Clone,Serialize,Deserialize)]
struct Trial{width_km:f64,center_km:f64,q:[f64;2],actual_h1400:f64,cost:f64,data_cost:f64,status:String,gradient:f64,iterations:usize,checkpoint:PathBuf}
pub fn refine(root:&Path,fit_path:&Path,coarse:&Path,out:&Path,target_index:usize)->Result<()>{
    ensure!(target_index<6,"invalid profile target");let target=[0.01,0.1,0.3,0.5,0.7,0.9][target_index];
    let fit:WindowFit=campaign::load(fit_path)?;let time=fit.start_utc_s;
    let output=out.join(format!("{time}-{target_index:02}.h5"));if output.exists(){return Ok(());}
    let reference:Reference=campaign::load(&root.join("references").join(format!("{time}.h5")))?;
    let bg:Background=campaign::load(&root.join("background.h5"))?;
    let gains:BTreeMap<String,f64>=campaign::load(&root.join("calibration.h5"))?;
    let mut prep=Vec::new();
    for b in &fit.beams{let d:Decoded=campaign::load(&root.join("products").join(format!("{}.h5",b.id)))?;
        let key=format!("{:.2}-{:.2}",b.azimuth_deg,b.elevation_deg);let mut p=prepare(d,&reference,&bg,gains[&key],&[])?;
        p.initial=project_initial(b,&p.spline)?;prep.push(p);}
    let refs=prep.iter().map(|p|&p.problem).collect::<Vec<_>>();
    let baseline=prep.iter().map(|p|p.initial.clone()).collect::<Vec<_>>();
    let work=out.join(format!("{time}-{target_index:02}"));std::fs::create_dir_all(&work)?;
    let max_width=maximum_width(&reference.shape,1400.,target);
    if max_width<WIDTH_MIN{campaign::save(&output,&serde_json::json!({"time":time,"target_h1400":target,"status":"outside_physical_transition_bounds"}))?;return Ok(());}
    let upper=if max_width<WIDTH_MAX{max_width-1e-6}else{WIDTH_MAX};
    let nominal=Shape::parameters([fit.shared_parameters[0],fit.shared_parameters[1]]).1;
    let mut anchors=vec![WIDTH_MIN,100f64.min(upper),nominal.clamp(WIDTH_MIN,upper),upper];
    anchors.sort_by(f64::total_cmp);anchors.dedup_by(|a,b|(*a-*b).abs()<1e-7);
    let mut trials=Vec::<Trial>::new();
    let mut evaluate=|width:f64|->Result<f64>{
        let width=(width.clamp(WIDTH_MIN,upper)*1e8).floor()/1e8;
        if let Some(old)=trials.iter().filter(|t|(t.width_km-width).abs()<1e-7&&t.status=="converged").min_by(|a,b|a.cost.total_cmp(&b.cost)){return Ok(old.cost);}
        let roots=transition_roots(&reference.shape,1400.,target,width);ensure!(!roots.is_empty(),"empty exact H constraint at width {width}");
        let mut best=f64::INFINITY;
        for q in roots{
            let center=Shape::parameters(q).0;let path=work.join(format!("w-{width:.8}-c-{center:.8}.h5"));
            let checkpoint=path.with_extension("checkpoint.h5");
            let trial=if path.exists(){campaign::load::<Trial>(&path)?}else{
                let mut initial=baseline.clone();let mut closest=f64::INFINITY;
                for t in &trials{if t.status!="converged"||!t.checkpoint.exists(){continue;}
                    let distance=(t.width_km/width).ln().abs()+(t.center_km-center).abs()/1450.;
                    if distance<closest{let cp:serde_json::Value=campaign::load(&t.checkpoint)?;initial=serde_json::from_value(cp["nodes"].clone())?;closest=distance;}}
                if !closest.is_finite(){for case in (1+target_index*3)..(4+target_index*3){let cp=coarse.join(format!("{time}-{case:02}.checkpoint.h5"));
                    if !cp.exists(){continue;}let value:serde_json::Value=campaign::load(&cp)?;let nodes:Vec<Vec<coupled::Node>>=serde_json::from_value(value["nodes"].clone())?;
                    let old_q=[nodes[0][0].h_fraction,nodes[0][1].h_fraction];let(c,w)=Shape::parameters(old_q);
                    let distance=(w/width).ln().abs()+(c-center).abs()/1450.;if distance<closest{initial=nodes;closest=distance;}}}
                ensure!(initial.len()==baseline.len()&&initial.iter().zip(&baseline).all(|(a,b)|a.len()==b.len()),"profile warm-start dimensions");
                let sol=sharedcomposition::solve_fixed_h(&refs,&initial,Arc::new(reference.shape.clone()),q,0,Some(&checkpoint))?;
                let actual=reference.shape.evaluate(1400.,[sol.shared[0],sol.shared[1]]).0;ensure!((actual-target).abs()<1e-10,"H constraint drift");
                let mut data=0.;for(p,n)in prep.iter().zip(&sol.nodes){let(r,_)=p.problem.evaluate(&coupled::encode(n),false)?;
                    data+=r.iter().take(p.problem.products.len()*2).map(|v|if v.abs()<=3.{v*v}else{6.*v.abs()-9.}).sum::<f64>();}
                let t=Trial{width_km:width,center_km:center,q,actual_h1400:actual,cost:sol.robust_objective,data_cost:data,status:sol.status,gradient:sol.normalized_gradient,iterations:sol.iterations,checkpoint};
                campaign::save(&path,&t)?;t};
            if trial.status=="converged"{best=best.min(trial.cost);}trials.push(trial);
        }Ok(best)
    };
    let mut costs=Vec::new();for w in &anchors{costs.push(evaluate(*w)?);}
    let best=costs.iter().enumerate().min_by(|a,b|a.1.total_cmp(b.1)).unwrap().0;
    let mut left=anchors[best.saturating_sub(1)].ln();let mut right=anchors[(best+1).min(anchors.len()-1)].ln();
    let ratio=(5f64.sqrt()-1.)/2.;let mut x=right-ratio*(right-left);let mut y=left+ratio*(right-left);
    let(mut fx,mut fy)=(evaluate(x.exp())?,evaluate(y.exp())?);let mut steps=0;
    while right-left>0.001&&steps<24{if fx<fy{right=y;y=x;fy=fx;x=right-ratio*(right-left);fx=evaluate(x.exp())?;}else{left=x;x=y;fx=fy;y=left+ratio*(right-left);fy=evaluate(y.exp())?;}steps+=1;}
    drop(evaluate);
    let selected=trials.iter().filter(|t|t.status=="converged").min_by(|a,b|a.cost.total_cmp(&b.cost)).ok_or_else(||anyhow::anyhow!("no converged exact-H trial"))?;
    let unresolved=trials.iter().filter(|t|t.status!="converged").count();
    campaign::save(&output,&serde_json::json!({"time":time,"target_h1400":target,"actual_h1400":selected.actual_h1400,"target_achieved":true,"status":if unresolved==0&&right-left<=0.001{"converged"}else{"unresolved_width_search"},"data_objective":selected.data_cost,"spatial_penalized_objective":selected.cost,"q":selected.q,"center_width_km":[selected.center_km,selected.width_km],"width_log_bracket":[left,right],"width_search_steps":steps,"unresolved_trials":unresolved,"trials":trials,"note":"Exact H fraction; all transition-height roots retained; width refined in the best coarse bracket; all beam Ne/Ti/Te/vi reoptimized. Temporal priors removed, altitude priors retained. Conditional working objective, not calibrated confidence coverage."}))?;
    Ok(())
}
#[cfg(test)]mod tests{
    use super::*;
    #[test]fn exact_fraction_including_multiple_transition_heights(){
        let s=Shape{altitude_km:vec![550.,1400.,2000.],scale_height_km:vec![50.,500.,600.],source:"test".into()};s.validate().unwrap();
        let roots=transition_roots(&s,1400.,0.72,300.);assert_eq!(roots.len(),2);
        for q in roots{assert!((s.evaluate(1400.,q).0-0.72).abs()<1e-12);}
        for f in [0.01,0.1,0.5,0.9]{for q in transition_roots(&s,1400.,f,20.){assert!((s.evaluate(1400.,q).0-f).abs()<1e-12);}}
    }
}

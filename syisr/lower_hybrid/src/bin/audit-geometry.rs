//! Independent native Rust IGRF-14/WGS84 check of the Python beam markers.
use anyhow::{ensure,Result};
use chrono::{Datelike,DateTime,Utc};
use hdf5::{File,types::VarLenUnicode};

#[path="../../../src/geomagnetic.rs"]
#[allow(dead_code)]
mod geomagnetic;

type V=[f64;3];
fn dot(a:V,b:V)->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn add(a:V,b:V,s:f64)->V{std::array::from_fn(|i|a[i]+s*b[i])}
fn basis(lat:f64,lon:f64)->[V;3]{
    let(p,l)=(lat.to_radians(),lon.to_radians());
    [[-l.sin(),l.cos(),0.],[-p.sin()*l.cos(),-p.sin()*l.sin(),p.cos()],[p.cos()*l.cos(),p.cos()*l.sin(),p.sin()]]
}
fn ecef(lat:f64,lon:f64,height:f64)->V{
    let(p,l)=(lat.to_radians(),lon.to_radians());let e2=6.6943799901413165e-3;
    let n=6378.137/(1.-e2*p.sin().powi(2)).sqrt();
    [(n+height)*p.cos()*l.cos(),(n+height)*p.cos()*l.sin(),(n*(1.-e2)+height)*p.sin()]
}
fn geod(x:V)->V{
    let e2=6.6943799901413165e-3;let r=x[0].hypot(x[1]);let mut p=x[2].atan2(r*(1.-e2));
    for _ in 0..15{let n=6378.137/(1.-e2*p.sin().powi(2)).sqrt();p=(x[2]+e2*n*p.sin()).atan2(r);}
    let n=6378.137/(1.-e2*p.sin().powi(2)).sqrt();[p.to_degrees(),x[1].atan2(x[0]).to_degrees(),r/p.cos()-n]
}
fn field(year:f64,x:V)->Result<V>{
    let q=geod(x);let ned=geomagnetic::igrf14(year,q[0],q[1],q[2])?;let enu=basis(q[0],q[1]);
    Ok(std::array::from_fn(|i|ned[0]*enu[1][i]+ned[1]*enu[0][i]-ned[2]*enu[2][i]))
}
fn main()->Result<()>{
    let a:Vec<_>=std::env::args().collect();ensure!(a.len()==2,"audit-geometry AUDIT.h5");
    let h=File::open_rw(&a[1])?;let site=h.dataset("site_lat_lon_deg")?.read_raw::<f64>()?;
    let origin=ecef(site[0],site[1],0.);let axes=basis(site[0],site[1]);
    let date=h.attr("geometry_epoch_utc")?.read_scalar::<VarLenUnicode>()?;
    let date=DateTime::parse_from_rfc3339(date.as_str())?.with_timezone(&Utc);
    let y=date.year();let jan=chrono::NaiveDate::from_ymd_opt(y,1,1).unwrap().and_hms_opt(0,0,0).unwrap().and_utc().timestamp();
    let next=chrono::NaiveDate::from_ymd_opt(y+1,1,1).unwrap().and_hms_opt(0,0,0).unwrap().and_utc().timestamp();
    let year=y as f64+(date.timestamp()-jan) as f64/(next-jan) as f64;
    let rows=h.dataset("beam_geometry")?.read_raw::<f64>()?;
    let reference=h.dataset("target_geodetic_and_field_ecef_nT")?.read_raw::<f64>()?;
    let mut results=Vec::new();
    for (i,row) in rows.chunks(7).enumerate(){
        let(az,el)=(row[1].to_radians(),row[2].to_radians());
        let u=std::array::from_fn(|k|az.sin()*el.cos()*axes[0][k]+az.cos()*el.cos()*axes[1][k]+el.sin()*axes[2][k]);
        let point=add(origin,u,row[3]);let b=field(year,point)?;
        let aspect=(dot(u,b)/dot(b,b).sqrt()).asin().to_degrees();
        let ref_b:[f64;3]=reference[i*6+3..i*6+6].try_into().unwrap();
        let delta=dot(add(b,ref_b,-1.),add(b,ref_b,-1.)).sqrt();
        let mut low=row[3]-100.;let mut high=row[3]+100.;
        let sign=dot(u,field(year,add(origin,u,low))?);
        ensure!(sign*dot(u,field(year,add(origin,u,high))?)<0.,"native field root not bracketed");
        for _ in 0..60{let mid=(low+high)/2.;if dot(u,field(year,add(origin,u,mid))?)*sign>0.{low=mid;}else{high=mid;}}
        let root=(low+high)/2.;
        ensure!(delta<3. && (root-row[3]).abs()<0.1,"native/Python geometry disagreement");
        println!("beam {i}: native range {root:.6} km, Python range {:.6} km, difference {:.6} km; field delta {delta:.6} nT, aspect {aspect:.8} deg",row[3],root-row[3]);
        results.extend([root,root-row[3],delta,aspect]);
    }
    if h.link_exists("native_rust_geometry_audit"){h.unlink("native_rust_geometry_audit")?;}
    h.new_dataset::<f64>().shape((5,4)).create("native_rust_geometry_audit")?.write_raw(&results)?;
    Ok(())
}

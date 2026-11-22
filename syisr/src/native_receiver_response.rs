//! Finite-range strong-AC ambiguity coefficients on the native receiver-tap grid.
//! The source is constant inside each 20 us range cell. Fractional tap/cell
//! boundaries are integrated exactly; no output-rate FIR approximation is made.
use anyhow::{ensure, Result};
use rayon::prelude::*;

pub struct Stencil {
    pub first_cell: i32,
    pub last_cell: i32,
    pub first_intrinsic_us: i32,
    pub last_intrinsic_us: i32,
    /// lag (1..=lags), source cell, intrinsic lag (integer microseconds).
    pub coefficients: Vec<f64>,
}

pub fn build(codes: &[Vec<f64>], taps: &[(i32, f64)], lags: usize,
             first_cell: i32, last_cell: i32) -> Result<Stencil> {
    ensure!(codes.len()==64 && codes.iter().all(|c|c.len()==32 && c.iter().all(|v|v.abs()==1.)), "expected 64x32 signs");
    ensure!(!taps.is_empty() && taps.iter().all(|(_,h)|h.is_finite()) && (1..=62).contains(&lags) && first_cell<=last_cell,"invalid stencil settings");
    let amin=taps.iter().map(|x|x.0).min().unwrap();
    let amax=taps.iter().map(|x|x.0).max().unwrap();
    let lo=10+amin-amax;let hi=lags as i32*10+amax-amin;
    let nu=(hi-lo+1) as usize;let nr=(last_cell-first_cell+1) as usize;
    let mut gram=vec![0.;1024*1024];
    gram.par_chunks_mut(1024).enumerate().for_each(|(p,row)|{
        for (q,v) in row.iter_mut().enumerate(){*v=codes.iter().map(|c|c[p/32]*c[p%32]*c[q/32]*c[q%32]).sum::<f64>()/64.;}
    });
    let blocks:Vec<Vec<f64>>=(1..=lags).into_par_iter().map(|lag|{
        let branches:Vec<_>=[(0,lag/2),(1,(lag+1)/2)].into_iter().filter(|(_,l)|*l>0).collect();
        let mut result=vec![0.;nr*nu];
        for &(phase,l) in &branches {
            let factor=(if lag%2==1{2.}else{1.})/(branches.len()*(32-l)) as f64;
            // Sum decoder baud products once. All tap/source positions then
            // need only a lookup instead of another 32-baud loop.
            let mut g=vec![0.;63*63];
            for x in -31i32..=31 {for y in -31i32..=31 {
                g[((x+31)*63+y+31) as usize]=(0..32-l).filter_map(|b|{
                    let i=b as i32+x;let j=b as i32+y;
                    if (0..32).contains(&i)&&(0..32).contains(&j){Some(gram[(b*32+b+l)*1024+(i*32+j) as usize])}else{None}
                }).sum();
            }}
            let lookup=|x:i32,y:i32|if (-31..=31).contains(&x)&&(-31..=31).contains(&y){g[((x+31)*63+y+31) as usize]}else{0.};
            for &(a,ha) in taps {for &(b,hb) in taps {
                let weight=ha*hb*factor;if weight==0.{continue;}
                let u=lag as i32*10+a-b;let ku=(u-lo) as usize;
                let ta=phase as i32*10-a;
                let tb=phase as i32*10+lag as i32*10-b;
                for cell in first_cell..=last_cell {
                    let start=20*cell-10;let end=start+20;
                    // Each floor changes once within this 20 us source cell.
                    let mut cuts=[start,end,start+(ta-start).rem_euclid(20),start+(tb-start).rem_euclid(20)];
                    cuts.sort_unstable();let mut overlap=0.;
                    for interval in cuts.windows(2){let width=interval[1]-interval[0];if width==0{continue;}
                        // Twice the midpoint avoids rounding at a code edge.
                        let mid2=interval[0]+interval[1];
                        let x=(2*ta-mid2).div_euclid(40);let y=(2*tb-mid2).div_euclid(40);
                        overlap+=width as f64/20.*lookup(x,y);
                    }
                    result[(cell-first_cell) as usize*nu+ku]+=weight*overlap;
                }
            }}
        }
        result
    }).collect();
    Ok(Stencil{first_cell,last_cell,first_intrinsic_us:lo,last_intrinsic_us:hi,coefficients:blocks.into_iter().flatten().collect()})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Complex as C;
    #[test]
    fn native_taps_match_explicit_finite_code_integral(){
        let codes=crate::decode::phase(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"),"/m32-reference.bc"))).unwrap();
        let taps=[(-1,0.2),(0,0.6),(2,0.2)];let k=build(&codes,&taps,3,-2,2).unwrap();let f=7000.;
        let nu=(k.last_intrinsic_us-k.first_intrinsic_us+1) as usize;
        for lag in [2,3] {for d in [-2,0,2] {
            let start=((lag-1)*5+(d+2) as usize)*nu;
            let got:C=k.coefficients[start..start+nu].iter().enumerate().map(|(j,c)|C::from_polar(*c,-std::f64::consts::TAU*f*(k.first_intrinsic_us+j as i32) as f64*1e-6)).sum();
            let bs:Vec<_>=[(0,lag/2),(1,(lag+1)/2)].into_iter().filter(|(_,l)|*l>0).collect();let mut expected=C::default();
            for &(phase,l) in &bs {for b in 0..32-l {for code in &codes {
                let value=crate::exact_response::coded_cell((20*b+10*phase) as f64*1e-6,lag as f64*1e-5,20e-6,&code.iter().map(|v|C::new(*v,0.)).collect::<Vec<_>>(),[(20*d-10) as f64*1e-6,(20*d+10) as f64*1e-6],&taps.iter().map(|(t,h)|(*t as f64*1e-6,C::new(*h,0.))).collect::<Vec<_>>(),&[f]).unwrap()[0];
                expected+=value*32.*code[b]*code[b+l]*(if lag%2==1{2.}else{1.})/(bs.len()*64*(32-l)) as f64;
            }}}
            assert!((got-expected).norm()<1e-10,"lag{lag} cell{d}: {got} != {expected}");
        }}
    }
}

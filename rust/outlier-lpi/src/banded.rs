//! Exact banded Hermitian WLS, including an optional global nuisance column.
//! Column scaling and a Schur complement preserve the dense solver's units.
//! No regularization, lag interpolation, or truncated factorization is applied.
use crate::C;
use anyhow::{ensure,Result};
struct Band {n:usize,w:usize,a:Vec<C>}
impl Band {
 fn new(n:usize,w:usize)->Self{Self{n,w,a:vec![C::default();n*(w+1)]}}
 fn lower(&self,i:usize,j:usize)->C{if i-j<=self.w{self.a[i*(self.w+1)+i-j]}else{C::default()}}
 fn at(&self,i:usize,j:usize)->C{if i>=j{self.lower(i,j)}else{self.lower(j,i).conj()}}
 fn set(&mut self,i:usize,j:usize,v:C){self.a[i*(self.w+1)+i-j]=v;}
 fn add(&mut self,i:usize,j:usize,v:C){self.a[i*(self.w+1)+i-j]+=v;}
 fn factor(&mut self)->Result<f64>{
  let mut min=f64::INFINITY;let mut max=0_f64;
  for i in 0..self.n{for j in i.saturating_sub(self.w)..=i{
   let mut v=self.lower(i,j);
   for k in i.saturating_sub(self.w)..j{v-=self.lower(i,k)*self.lower(j,k).conj();}
   if i==j{ensure!(v.re.is_finite()&&v.re>0.&&v.im.abs()<1e-9*(1.+v.re.abs()),"nonpositive banded lag design");min=min.min(v.re);max=max.max(v.re);self.set(i,j,C::new(v.re.sqrt(),0.));}
   else{self.set(i,j,v/self.lower(j,j).re);}
  }}
  let ratio=min/max;ensure!(ratio>1e-12,"ill-conditioned banded lag design");Ok(ratio)
 }
 fn solve(&self,b:&[C])->Vec<C>{
  let mut x=b.to_vec();
  for i in 0..self.n{let mut y=x[i];for j in i.saturating_sub(self.w)..i{y-=self.lower(i,j)*x[j];}x[i]=y/self.lower(i,i).re;}
  for i in (0..self.n).rev(){let mut y=x[i];for j in i+1..(i+self.w+1).min(self.n){y-=self.lower(j,i).conj()*x[j];}x[i]=y/self.lower(i,i).re;}x
 }
 fn inverse_diagonal(&self)->Result<Vec<f64>>{
  // Selected-inverse recursion. Entries needed for each diagonal remain inside
  // the same band, so this does not construct a dense inverse.
  let mut s=Self::new(self.n,self.w);let mut diagonal=vec![0.;self.n];
  for i in (0..self.n).rev(){let d=self.lower(i,i).re;
   for j in i+1..(i+self.w+1).min(self.n){let mut v=C::default();for k in i+1..(i+self.w+1).min(self.n){v+=self.lower(k,i).conj()*s.at(k,j);}s.set(j,i,(-v/d).conj());}
   let mut v=C::new(1./(d*d),0.);for k in i+1..(i+self.w+1).min(self.n){v-=self.lower(k,i).conj()*s.at(k,i)/d;}
   ensure!(v.re.is_finite()&&v.re>0.&&v.im.abs()<1e-8*(1.+v.re.abs()),"invalid selected covariance");s.set(i,i,C::new(v.re,0.));diagonal[i]=v.re;
  }Ok(diagonal)
 }
}
/// `background_column` means the final column is a global nuisance coefficient.
/// Every other column must have finite local support. Variances are exact
/// diagonal entries of the conditional WLS covariance, not a diagonal-Fisher
/// approximation. Unconstrained columns remain NaN.
pub fn solve_banded(rows:&[Vec<(usize,C)>],y:&[C],sigma:&[f64],n:usize,background_column:bool)->Result<(Vec<C>,Vec<f64>,f64)>{
 ensure!(n>0&&rows.len()==y.len()&&rows.len()==sigma.len(),"banded WLS dimensions");
 let nr=n-usize::from(background_column);let mut scale=vec![0_f64;n];
 for (row,&s) in rows.iter().zip(sigma){ensure!(s.is_finite()&&s>0.,"nonpositive row sigma");for &(j,a)in row{ensure!(j<n&&a.re.is_finite()&&a.im.is_finite(),"invalid design entry");scale[j]+=a.norm_sqr()/(s*s);}}
 for s in &mut scale{*s=s.sqrt();}
 let active:Vec<_>=(0..nr).filter(|&j|scale[j]>0.).collect();let mut map=vec![usize::MAX;nr];for(i,&j)in active.iter().enumerate(){map[j]=i;}
 let background=background_column&&scale[n-1]>0.;ensure!(!active.is_empty()||background,"unconstrained design");
 let mut width=0;for row in rows{let indices:Vec<_>=row.iter().filter_map(|(j,_)|(*j<nr&&scale[*j]>0.).then(||map[*j])).collect();if let(Some(lo),Some(hi))=(indices.iter().min(),indices.iter().max()){width=width.max(hi-lo);}}
 let mut a=Band::new(active.len(),width);let mut rhs=vec![C::default();active.len()];let mut cross=rhs.clone();let mut rhs_b=C::default();let mut diagonal_b=0.;
 for((row,&value),&s)in rows.iter().zip(y).zip(sigma){ensure!(value.re.is_finite()&&value.im.is_finite(),"nonfinite measurement");
  let mut entries=vec![];let mut b=C::default();
  for &(j,z)in row{if scale[j]>0.{let z=z/(s*scale[j]);if j<nr{entries.push((map[j],z));}else{b+=z;}}}
  for &(i,z)in &entries{rhs[i]+=z.conj()*value/s;cross[i]+=z.conj()*b;for &(j,w)in &entries{if i>=j{a.add(i,j,z.conj()*w);}}}
  rhs_b+=b.conj()*value/s;diagonal_b+=b.norm_sqr();
 }
 let mut answer=vec![C::new(f64::NAN,f64::NAN);n];let mut variance=vec![f64::NAN;n];
 if active.is_empty(){ensure!(diagonal_b>0.,"empty background design");answer[n-1]=rhs_b/diagonal_b/scale[n-1];variance[n-1]=1./diagonal_b/scale[n-1].powi(2);return Ok((answer,variance,1.));}
 let mut condition=a.factor()?;let mut x=a.solve(&rhs);let mut diagonal=a.inverse_diagonal()?;
 if background{let v=a.solve(&cross);let schur=diagonal_b-cross.iter().zip(&v).map(|(b,v)|b.conj()*v).sum::<C>().re;ensure!(schur.is_finite()&&schur>1e-12*diagonal_b,"rank-deficient global background");condition=condition.min(schur/diagonal_b);
  let b=(rhs_b-cross.iter().zip(&x).map(|(b,x)|b.conj()*x).sum::<C>())/schur;
  for i in 0..x.len(){x[i]-=v[i]*b;diagonal[i]+=v[i].norm_sqr()/schur;}
  answer[n-1]=b/scale[n-1];variance[n-1]=1./schur/scale[n-1].powi(2);
 }
 for(i,&j)in active.iter().enumerate(){answer[j]=x[i]/scale[j];variance[j]=diagonal[i]/scale[j].powi(2);}
 Ok((answer,variance,condition))
}
#[cfg(test)]mod tests{
 use super::*;
 #[test]fn matches_dense_with_global_background_and_missing_column(){
  let n=17;let truth:Vec<_>=(0..n).map(|i|C::new((i as f64*0.7).sin(),(i as f64*0.2).cos())).collect();
  let mut rows=vec![];for k in 0..110{let mut row=vec![];for j in (k%14)..(k%14+3){if j!=7{row.push((j,C::from_polar(0.4+((k*19+j*7)%13)as f64,0.23*(k+j+j*k*3)as f64)));}}row.push((16,C::new(1.,0.)));rows.push(row);}
  let y:Vec<_>=rows.iter().map(|row|row.iter().map(|(j,a)|truth[*j]*a).sum()).collect();let sigma:Vec<_>=(0..rows.len()).map(|i|1.+(i%3)as f64).collect();
  let dense=crate::solve(&rows,&y,&sigma,n).unwrap();let band=solve_banded(&rows,&y,&sigma,n,true).unwrap();
  for j in 0..n{if dense.0[j].re.is_nan(){assert!(band.0[j].re.is_nan()&&band.1[j].is_nan());}else{assert!((dense.0[j]-band.0[j]).norm()<1e-10);assert!((dense.1[j]-band.1[j]).abs()<1e-10);}}
 }
 #[test]fn singular_background_is_rejected(){let rows=vec![vec![(0,C::new(1.,0.)),(1,C::new(1.,0.))];5];assert!(solve_banded(&rows,&[C::new(1.,0.);5],&[1.;5],2,true).is_err());}
}

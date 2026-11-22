// Stationary magnetized Ornstein-Uhlenbeck/Fokker-Planck displacement model.
// Constant friction/diffusion is the Brownian Coulomb approximation used in
// Kudeki & Milla (2011) / Milla et al. (2023), not a speed-dependent Landau operator.
// A(t)=integral_0^t (t-u) exp(-z u) du; its ballistic limit is t^2/2.
// At exact perpendicularity the electron transverse displacement has a rapidly
// converging double exponential series. Its causal transform is analytic and
// avoids resolving millions of gyroperiods in a low-density, narrow line.
fn perpendicular_electron_response(f:&[f64],n:f64,t:f64,m:f64,nu:f64,neutral:f64,b:f64,k:f64)->Vec<(C,f64)>{
    let sigma2=KB*t/m;let z=C::new(nu,-E*b/m);
    let q=C::new(sigma2*k*k,0.)/(z*z);
    assert!(q.norm()<2.,"perpendicular electron series outside tested k*rho domain");
    let diffusion=sigma2*k*k*(C::new(1.,0.)/z).re;
    let mut coefficients=vec![C::new(1.,0.);14];
    for j in 1..coefficients.len(){coefficients[j]=coefficients[j-1]*(-q/2.)/j as f64;}
    let aa=n*E*E/(EPS*KB*t*k*k);
    f.iter().map(|freq|{
        let omega=2.*PI*freq;let mut g=C::default();
        for (a,ca) in coefficients.iter().enumerate(){for (b,cb) in coefficients.iter().enumerate(){
            g+=*ca*cb.conj()/(C::new(diffusion+neutral,-omega)+a as f64*z+b as f64*z.conj());
        }}
        g*=q.re.exp();let j=g/(C::new(1.,0.)-neutral*g);
        (aa*(C::new(1.,0.)+C::new(0.,omega)*j),2.*j.re)
    }).collect()
}
fn ou_integral(z:C,t:f64)->C{
    let x=z*t;
    if x.norm()<0.05{
        let mut term=C::new(t*t/2.,0.);let mut sum=term;
        for k in 1..14{term*=-x/(k+2) as f64;sum+=term;}sum
    }else{(x-C::new(1.,0.)+(-x).exp())/(z*z)}
}
fn coulomb_response(f:&[f64],n:f64,t:f64,m:f64,nupar:f64,nuperp:f64,nuneutral:f64,b:f64,alpha:f64,refine:usize)->Vec<(C,f64)>{
    if alpha==0. && m<1e-29 && b>0. && (nuperp>0. || nuneutral>0.) && KB*t/m*K*K/(nuperp*nuperp+(E*b/m).powi(2))<2. {return perpendicular_electron_response(f,n,t,m,nuperp,nuneutral,b,K);}
    let sigma2=KB*t/m;let gyro=E*b/m;let kp=K*alpha.to_radians().sin();let kt=K*alpha.to_radians().cos();
    let df=f[1]-f[0];let mut duration=1./df;
    // A small endpoint can lie between gyroperiod recurrences. Bound the
    // entire late-time envelope before truncation, including collisionless ions.
    let zp=C::new(nuperp,-gyro);
    for _ in 0..12{
        let parallel=ou_integral(C::new(nupar,0.),duration).re;
        let perpendicular=((zp*duration-C::new(1.,0.))/(zp*zp)).re
            -(-nuperp*duration).exp()/zp.norm_sqr();
        let envelope=(-sigma2*(kp*kp*parallel+kt*kt*perpendicular)-nuneutral*duration).exp();
        if envelope<1e-12{break;}duration*=2.;
    }
    let target_dt=if m<1e-29{0.05e-6}else{0.5e-6};
    let count=(((duration/target_dt).ceil() as usize).next_power_of_two())*refine;
    let dt=duration/count as f64;
    let mut kernel:Vec<C>=(0..count).map(|i|{
        let time=i as f64*dt;
        let a=ou_integral(C::new(nupar,0.),time).re;
        let q=ou_integral(C::new(nuperp,-gyro),time).re;
        C::new((-sigma2*(kp*kp*a+kt*kt*q).max(0.)-nuneutral*time).exp(),0.)
    }).collect();
    assert!(kernel[count-1].re<1e-8,"Gordeyev time window too short: tail {}",kernel[count-1].re);
    let mut planner=rustfft::FftPlanner::<f64>::new();planner.plan_fft_inverse(count).process(&mut kernel);
    let aa=n*E*E/(EPS*KB*t*K*K);
    f.iter().map(|&freq|{
        let index=((freq*duration).round() as i64).rem_euclid(count as i64) as usize;
        let w=2.*PI*freq;
        // Euler-Maclaurin endpoint correction removes the leading trapezoid
        // bias in the causal transform; the neutral BGK rate is conserving.
        let derivative=C::new(-nuneutral,w);
        let third=derivative.powu(3)-3.*derivative*K*K*sigma2
            +C::new(sigma2*(kp*kp*nupar+kt*kt*nuperp),0.);
        let g=kernel[index]*dt-C::new(dt/2.,0.)+dt*dt/12.*derivative-dt.powi(4)/720.*third;
        let j=g/(C::new(1.,0.)-nuneutral*g);
        let free=2.*j.re;
        assert!(free>-1e-12,"Negative single-particle spectrum {}",free);
        (aa*(C::new(1.,0.)+C::new(0.,w)*j),free.max(0.))
    }).collect()
}

"""Joint source-height/aspect quadrature and intrinsic-lag Doppler response."""
__version__ = "0.2.0"
import math
import numpy as np

def geodetic_height_km(xyz):
    xyz=np.asarray(xyz);rho=np.hypot(xyz[...,0],xyz[...,1]);e2=6.6943799901413165e-3
    lat=np.arctan2(xyz[...,2],rho*(1-e2))
    for _ in range(10):
        n=6378.137/np.sqrt(1-e2*np.sin(lat)**2)
        lat=np.arctan2(xyz[...,2]+e2*n*np.sin(lat),rho)
    return rho/np.cos(lat)-6378.137/np.sqrt(1-e2*np.sin(lat)**2)

def joint_height_aspect(height,aspect,angular,height_nodes,aspect_nodes):
    """Signed source geometry is converted to |aspect| only for even ISR theory.

    Return source-cell x height-node x aspect-node weights, plus omitted mass.
    No endpoint extension or renormalization is performed.
    """
    height=np.asarray(height);aa=abs(np.asarray(aspect));angular=np.asarray(angular)
    z=np.asarray(height_nodes);a=np.asarray(aspect_nodes)
    if height.shape!=aa.shape or height.shape[-1]!=len(angular):raise ValueError('Quadrature shape mismatch')
    if np.any(np.diff(z)<=0) or np.any(np.diff(a)<=0) or np.any(angular<0):raise ValueError('Invalid quadrature')
    joint=np.zeros((height.shape[0],len(z),len(a)));outside_z=[];outside_a=[]
    for s in range(len(height)):
        inz=(height[s]>=z[0])&(height[s]<=z[-1]);ina=aa[s]<=a[-1];valid=inz&ina
        outside_z.append(angular[~inz].sum());outside_a.append(angular[inz&~ina].sum())
        zh=np.clip(np.searchsorted(z,height[s,valid]),1,len(z)-1);ah=np.clip(np.searchsorted(a,aa[s,valid]),1,len(a)-1)
        zw=(height[s,valid]-z[zh-1])/(z[zh]-z[zh-1]);aw=(aa[s,valid]-a[ah-1])/(a[ah]-a[ah-1]);ww=angular[valid]
        for zi,zv in [(zh-1,1-zw),(zh,zw)]:
            for ai,av in [(ah-1,1-aw),(ah,aw)]:
                joint[s]+=np.bincount(zi*len(a)+ai,weights=ww*zv*av,minlength=len(z)*len(a)).reshape(len(z),len(a))
    np.testing.assert_allclose(joint.sum((1,2))+outside_z+np.asarray(outside_a),angular.sum(),atol=1e-12)
    return joint,np.asarray(outside_z),np.asarray(outside_a)

def doppler_moments(intrinsic_lag_s,maximum_hz=1000.,tolerance=1e-10):
    """Taylor coefficients for exp(i2pi f u), with a uniform remainder bound."""
    u=np.asarray(intrinsic_lag_s);x=2*np.pi*maximum_hz*max(abs(u));order=0
    while math.exp(x)*x**(order+1)/math.factorial(order+1)>tolerance:
        order+=1
        if order>100:raise ValueError('Doppler expansion exceeds supported order')
    m=np.empty((order+1,len(u)),complex);m[0]=1
    for j in range(1,len(m)):m[j]=m[j-1]*(2j*np.pi*maximum_hz*u)/j
    return m,math.exp(x)*x**(order+1)/math.factorial(order+1)

def doppler_weights(frequency_hz,maximum_hz,count):
    if abs(frequency_hz)>maximum_hz*(1+1e-12):raise ValueError('Doppler outside validated expansion')
    x=frequency_hz/maximum_hz
    w=x**np.arange(count);d=np.zeros(count);d[1:]=np.arange(1,count)*x**np.arange(count-1)/maximum_hz
    return w,d

def source_response_moments(kernel,joint,source_knot,intrinsic_acf,source_range_km,moments):
    intrinsic=(joint[:,source_knot,:]@intrinsic_acf)/np.asarray(source_range_km)[:,None]**2
    response_lu=np.einsum('lsu,su->lu',kernel,intrinsic,optimize=True)
    return moments@response_lu.T

if __name__=='__main__':
    # A height/aspect correlation must survive the angular average.
    h=np.array([[200.,300.]]);a=np.array([[0.,1.]]);w=np.array([.25,.75])
    joint,oz,oa=joint_height_aspect(h,a,w,[200.,300.],[0.,1.])
    np.testing.assert_allclose(joint[0],[[.25,0],[0,.75]])
    assert not np.allclose(joint[0],np.outer(joint[0].sum(1),joint[0].sum(0)))
    j,oz,oa=joint_height_aspect(np.array([[100.,350.]]),a,w,[200.,300.],[0.,1.])
    assert j.sum()==0 and oz[0]==1 # no artificial boundary extension
    u=np.linspace(-.0012,.0012,241);m,bound=doppler_moments(u)
    for f in np.linspace(-1000,1000,21):
        p,dp=doppler_weights(f,1000,len(m));actual=p@m;truth=np.exp(2j*np.pi*f*u)
        assert np.max(abs(actual-truth))<bound+2e-12
        np.testing.assert_allclose(dp@m,2j*np.pi*u*truth,atol=1e-11)
    for lat in [0,18.34,70.]:
        e2=6.6943799901413165e-3;p=np.deg2rad(lat);n=6378.137/np.sqrt(1-e2*np.sin(p)**2)
        for z in [80.,300.,650.]:
            xyz=[(n+z)*np.cos(p),0.,(n*(1-e2)+z)*np.sin(p)]
            np.testing.assert_allclose(geodetic_height_km(xyz),z,atol=1e-8)
    rng=np.random.default_rng(913);kernel=rng.normal(size=(3,2,len(u)))
    heights=np.array([[200.,250.,300.],[210.,260.,290.]]);angles=np.array([[0.,.5,1.],[.2,.7,.9]]);angular=np.array([.2,.3,.5]);ranges=np.array([250.,260.])
    joint,_,_=joint_height_aspect(heights,angles,angular,[200.,300.],[0.,1.])
    profiles=np.array([np.exp(-(u[None,:]/np.array([[.0004],[.00015]]))**2),np.exp(-(u[None,:]/np.array([[.0007],[.0003]]))**2)])
    result=np.zeros(3,complex);direct=np.zeros(3,complex)
    for k,(density,frequency) in enumerate([(1.,120.),(3.,-240.)]):
        coefficients=source_response_moments(kernel,joint,k,profiles[k],ranges,m)
        weights,_=doppler_weights(frequency,1000,len(m));result+=density*(weights@coefficients)
        intrinsic=(joint[:,k]@profiles[k])*np.exp(2j*np.pi*frequency*u)[None,:]/ranges[:,None]**2
        direct+=density*np.einsum('lsu,su->l',kernel,intrinsic)
    np.testing.assert_allclose(result,direct,rtol=1e-10,atol=1e-12)
    print('PASS joint geometry, partition, unclamped boundaries, WGS84 and intrinsic Doppler/value derivative')

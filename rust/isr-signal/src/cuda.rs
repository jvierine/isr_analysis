//! Explicit, reusable CUDA backend. Upload a pulse batch once for all requested
//! lags. Uses driver-loaded PTX: no CUDA toolkit or runtime compiler required.
use num_complex::Complex64 as C;
use cudarc::{driver::{CudaContext,CudaFunction,LaunchConfig,PushKernelArg},nvrtc::Ptx};
use std::sync::Arc;
type Error = Box<dyn std::error::Error + Send + Sync>;
pub struct CudaLagProducts { context: Arc<CudaContext>, function: CudaFunction, budget_bytes: usize }
impl CudaLagProducts {
    pub fn new(device:usize,budget_bytes:usize)->Result<Self,Error> {
        let context=CudaContext::new(device)?;
        let module=context.load_module(Ptx::from_src(include_str!("lag.ptx")))?;
        let function=module.load_function("lag_products")?;
        Ok(Self{context,function,budget_bytes})
    }
    /// Flat input is pulse-major. Each returned lag is pulse-major, then gate.
    /// No automatic precision reduction; outputs preserve signed complex f64.
    pub fn products(&self,z:&[C],pulse_len:usize,lags:&[usize],d:usize)->Result<Vec<Vec<C>>,Error>{
        if pulse_len==0 || z.is_empty() || z.len()%pulse_len!=0 || d==0 || lags.iter().any(|&l|l>=pulse_len) {
            return Err("invalid pulse batch, lag or decimation".into());
        }
        // Device holds input plus one lag output; caller must chunk larger input.
        if z.len().checked_mul(32).ok_or("batch size overflow")?>self.budget_bytes {return Err("GPU batch exceeds explicit memory budget".into());}
        let stream=self.context.default_stream();
        let flat:Vec<f64>=z.iter().flat_map(|v|[v.re,v.im]).collect();
        let input=stream.clone_htod(&flat)?;
        let mut result=Vec::with_capacity(lags.len());
        for &lag in lags {
            let gates=(pulse_len-lag)/d;
            let count=gates*(z.len()/pulse_len);
            if count==0 {result.push(vec![]);continue;}
            let count32=u32::try_from(count.checked_mul(32).ok_or("thread count overflow")?).map_err(|_|"too many output gates")?;
            let mut output=stream.alloc_zeros::<f64>(2*count)?;
            let (len,l,dec,g,n)=(pulse_len as u64,lag as u64,d as u64,gates as u64,count as u64);
            let mut args=stream.launch_builder(&self.function);
            args.arg(&input).arg(&mut output).arg(&len).arg(&l).arg(&dec).arg(&g).arg(&n);
            // Kernel bounds each thread; layout and buffer sizes checked above.
            unsafe{args.launch(LaunchConfig::for_num_elems(count32))?;}
            let host=stream.clone_dtoh(&output)?;
            result.push(host.chunks_exact(2).map(|v|C::new(v[0],v[1])).collect());
        }
        Ok(result)
    }
}

// Native Q,I int32 -> Hann-windowed, positive-phase-positive-frequency STFT.
// Deliberately no alternating-code decoding until the 20261125 sequence is verified.
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
static thread_local char error_text[256];
#define CUDA(x) do {cudaError_t e=(x);if(e!=cudaSuccess){snprintf(error_text,256,"%s",cudaGetErrorString(e));return 1;}}while(0)
#define FFT(x) do {cufftResult e=(x);if(e!=CUFFT_SUCCESS){snprintf(error_text,256,"cuFFT error %d",int(e));return 2;}}while(0)
struct Context {int n, batch; int32_t* input; cufftComplex* voltage; float* power; cufftHandle plan;};
__global__ void prepare(const int32_t* in,cufftComplex* out,int n,int total){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=total)return;
 float w=0.5f-0.5f*cosf(6.283185307179586f*(k%n)/(n-1));
 out[k]=make_cuFloatComplex(float(in[2*k+1])*w,float(in[2*k])*w);
}
__global__ void powers(const cufftComplex* x,float* out,int n,int total){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=total)return;
 // Periodic sum for a symmetric Hann is exactly 3*(n-1)/8.
 auto v=x[(k/n)*n+(k%n+n/2)%n];out[k]=(v.x*v.x+v.y*v.y)/(3.f*(n-1)/8.f);
}
extern "C" const char* pl_error(){return error_text;}
extern "C" int pl_create(int device,int n,int batch,void** result){
 CUDA(cudaSetDevice(device));auto c=new Context{};c->n=n;c->batch=batch;
 CUDA(cudaMalloc(&c->input,size_t(n)*batch*8));CUDA(cudaMalloc(&c->voltage,size_t(n)*batch*8));CUDA(cudaMalloc(&c->power,size_t(n)*batch*4));
 FFT(cufftPlan1d(&c->plan,n,CUFFT_C2C,batch));*result=c;return 0;
}
extern "C" int pl_power(void* ptr,const int32_t* input,float* output){
 auto c=(Context*)ptr;int total=c->n*c->batch;
 CUDA(cudaMemcpy(c->input,input,size_t(total)*8,cudaMemcpyHostToDevice));
 prepare<<<(total+255)/256,256>>>(c->input,c->voltage,c->n,total);CUDA(cudaGetLastError());
 FFT(cufftExecC2C(c->plan,c->voltage,c->voltage,CUFFT_FORWARD));
 powers<<<(total+255)/256,256>>>(c->voltage,c->power,c->n,total);CUDA(cudaGetLastError());
 CUDA(cudaMemcpy(output,c->power,size_t(total)*4,cudaMemcpyDeviceToHost));return 0;
}
extern "C" void pl_destroy(void* ptr){auto c=(Context*)ptr;if(!c)return;cufftDestroy(c->plan);cudaFree(c->input);cudaFree(c->voltage);cudaFree(c->power);delete c;}

// Adapted sliding conjugate-template/echo FFT architecture from Nate Hilliard's
// PlasmaLine_CUDA (CUDA 7-era archive supplied by the user). Bounds checks,
// persistent plans, explicit normalization and voltage whitening are new.
struct Match {int ne,nt,nfft,nr,step,nout,nwhite;int32_t* qi;float *gain,*codes,*out;cufftComplex *echo,*batch;cufftHandle whiteplan,plan;};
__global__ void unpack(const int32_t* qi,cufftComplex* echo,int ne,int nw){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<nw)echo[k]=k<ne?make_cuFloatComplex(qi[2*k+1],qi[2*k]):make_cuFloatComplex(0,0);}
__global__ void apply_gain(cufftComplex* x,const float* gain,int n){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<n){x[k].x*=gain[k]/n;x[k].y*=gain[k]/n;}}
__global__ void code_echo(const cufftComplex* echo,const float* codes,cufftComplex* batch,int phase,int nt,int nf,int nr,int step){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nf*nr)return;int r=k/nf,j=k%nf;
 if(j>=nt){batch[k]=make_cuFloatComplex(0,0);return;}auto v=echo[r*step+j];float c=codes[phase*nt+j];batch[k]=make_cuFloatComplex(v.x*c,v.y*c);
}
__global__ void reduce_power(const cufftComplex* x,float* out,int nf,int no,int nr,int nt){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=nr*no)return;int r=k/no,j=k%no,factor=nf/no;float s=0;
 for(int i=0;i<factor;i++){auto v=x[r*nf+(j*factor+i+nf/2)%nf];s+=v.x*v.x+v.y*v.y;}out[k]=s/(nt*factor);
}
extern "C" int pl_match_create(int device,int ne,int nt,int nf,int step,int no,const float* codes,const float* gain,void** result){
 CUDA(cudaSetDevice(device));auto c=new Match{};c->ne=ne;c->nt=nt;c->nfft=nf;c->step=step;c->nout=no;c->nr=(ne-nt)/step+1;c->nwhite=1;while(c->nwhite<ne)c->nwhite*=2;
 if(nt>nf||nf%no||c->nr<1){snprintf(error_text,256,"invalid match dimensions");return 3;}
 CUDA(cudaMalloc(&c->qi,size_t(ne)*8));CUDA(cudaMalloc(&c->echo,size_t(c->nwhite)*8));CUDA(cudaMalloc(&c->batch,size_t(nf)*c->nr*8));CUDA(cudaMalloc(&c->out,size_t(no)*c->nr*4));CUDA(cudaMalloc(&c->gain,size_t(c->nwhite)*4));CUDA(cudaMalloc(&c->codes,size_t(nt)*64*4));
 CUDA(cudaMemcpy(c->gain,gain,size_t(c->nwhite)*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(c->codes,codes,size_t(nt)*64*4,cudaMemcpyHostToDevice));
 FFT(cufftPlan1d(&c->whiteplan,c->nwhite,CUFFT_C2C,1));FFT(cufftPlan1d(&c->plan,nf,CUFFT_C2C,c->nr));*result=c;return 0;
}
extern "C" int pl_match_power(void* ptr,const int32_t* qi,int phase,float* out){
 auto c=(Match*)ptr;if(phase<0||phase>=64){snprintf(error_text,256,"invalid code phase");return 3;}
 CUDA(cudaMemcpy(c->qi,qi,size_t(c->ne)*8,cudaMemcpyHostToDevice));unpack<<<(c->nwhite+255)/256,256>>>(c->qi,c->echo,c->ne,c->nwhite);CUDA(cudaGetLastError());
 FFT(cufftExecC2C(c->whiteplan,c->echo,c->echo,CUFFT_FORWARD));apply_gain<<<(c->nwhite+255)/256,256>>>(c->echo,c->gain,c->nwhite);CUDA(cudaGetLastError());FFT(cufftExecC2C(c->whiteplan,c->echo,c->echo,CUFFT_INVERSE));
 int total=c->nfft*c->nr;code_echo<<<(total+255)/256,256>>>(c->echo,c->codes,c->batch,phase,c->nt,c->nfft,c->nr,c->step);CUDA(cudaGetLastError());FFT(cufftExecC2C(c->plan,c->batch,c->batch,CUFFT_FORWARD));
 total=c->nout*c->nr;reduce_power<<<(total+255)/256,256>>>(c->batch,c->out,c->nfft,c->nout,c->nr,c->nt);CUDA(cudaGetLastError());CUDA(cudaMemcpy(out,c->out,size_t(total)*4,cudaMemcpyDeviceToHost));return 0;
}
extern "C" void pl_match_destroy(void* ptr){auto c=(Match*)ptr;if(!c)return;cufftDestroy(c->whiteplan);cufftDestroy(c->plan);cudaFree(c->qi);cudaFree(c->echo);cudaFree(c->batch);cudaFree(c->out);cudaFree(c->gain);cudaFree(c->codes);delete c;}

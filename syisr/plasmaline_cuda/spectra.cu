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

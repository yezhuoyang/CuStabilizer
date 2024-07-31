#include <cstdio> 
#include <vector>
#include <iostream>
#include <cuda_runtime.h>



__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}


__global__  void vector_add(float *A,float *B, float *C, int N) {
     int i=blockDim.x*blockIdx.x+threadIdx.x;
     if(i<N){
        C[i]=A[i]+B[i];
     }
}




int main() {


  int N=100;
  size_t size=N* sizeof(float);

  //Allocate vectors in host memory
  float* h_A =(float*) malloc(size);
  float* h_B =(float*) malloc(size); 
  float* h_C =(float*) malloc(size);

  //Initialize input vector
  for(int i=0;i<N;++i){
      h_A[i]=i;
      h_B[i]=i;
      h_C[i]=i;
  }

  float* d_A;
  cudaMalloc(&d_A,size);
  float* d_B;
  cudaMalloc(&d_B,size);
  float* d_C;
  cudaMalloc(&d_C,size);    

  cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);  
  cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
  cudaMemcpy(d_C,h_C,size,cudaMemcpyHostToDevice);
    

  int threadsPerBlock = 256;
  int blocksPerGrid = (N+threadsPerBlock-1)/threadsPerBlock;

  vector_add<<<blocksPerGrid,threadsPerBlock>>>(d_A,d_B,d_C,N);

 //copy result from device memory to host memory
 //h_C contains the result in host memory
 
 cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);

 cudaFree(d_A);
 cudaFree(d_B);
 cudaFree(d_C);

  //Initialize input vector
 for(int i=0;i<N;++i){
      std::cout<<h_C[i]<<std::endl;   
  }
 
 return 0;
}
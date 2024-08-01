#include <cstdio> 
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <bitset>
#include <sstream>
#include <fstream>




#define getTableauElement(tableauMatrix,rowsize, row, col) (tableauMatrix[row*rowsize+col/8]&(0b10000000>>(col%8)))>>(7-(col%8))


#define setTableauValue(tableauMatrix,rowsize, row, col,value)       \
    do {                             \
        tableauMatrix[row*rowsize+col/8]=(value==1?(tableauMatrix[row*rowsize+col/8]|(0b00000000^(1<<(7-(col%8))))):(tableauMatrix[row*rowsize+col/8]&(0b11111111^(1<<(7-(col%8)))))); \
    } while (0)




__global__ void P_cuda(unsigned char* tableauMatrix,size_t target,int qubit_num,int rowsize,int N){
    int row=threadIdx.x;
    if(row<N){
        int r=getTableauElement(tableauMatrix,rowsize, row, (2*qubit_num));
        int zi=getTableauElement(tableauMatrix,rowsize, row, target);
        int xi=getTableauElement(tableauMatrix,rowsize, row, (target+qubit_num));
        setTableauValue(tableauMatrix,rowsize, row, (2*qubit_num),((r+zi*xi)%2));
        setTableauValue(tableauMatrix,rowsize, row, target,((zi+xi)%2));
    }
}

__global__ void H_cuda(unsigned char* tableauMatrix,size_t target,int qubit_num,int rowsize,int N){
    int row=threadIdx.x;
    if(row<N){
        int tmp=getTableauElement(tableauMatrix,rowsize, row, target);
        int z=getTableauElement(tableauMatrix,rowsize, row, (target+qubit_num));
        int r=getTableauElement(tableauMatrix,rowsize, row, (2*qubit_num));
        setTableauValue(tableauMatrix,rowsize, row, (2*qubit_num),((r+tmp*z)%2));
        setTableauValue(tableauMatrix,rowsize, row, target,z);
        setTableauValue(tableauMatrix,rowsize, row, (target+qubit_num),tmp);
    }
}


__global__ void CNOT_cuda(int *tableauMatrix,size_t control,size_t target,int qubit_num,int rowsize,int N){
     int row=threadIdx.x;
    if(row<N){
        int zi=getTableauElement(tableauMatrix,rowsize, row, control);
        int xi=getTableauElement(tableauMatrix,rowsize, row, (control+qubit_num));
        int zj=getTableauElement(tableauMatrix,rowsize, row, target);
        int xj=getTableauElement(tableauMatrix,rowsize, row, (target+qubit_num));
        int r=getTableauElement(tableauMatrix,rowsize, row, (2*qubit_num));

        int multi=(zj*xi);
        int xorsum=(xj+zi+1)%2;

        setTableauValue(tableauMatrix,rowsize, row, (2*qubit_num),((r+multi+xorsum)%2));
        setTableauValue(tableauMatrix,rowsize, row, (target+qubit_num),((xi+xj)%2));
        setTableauValue(tableauMatrix,rowsize, row, control,((zi+zj)%2));
    }   
}


// Function to print the binary representation of a char
void printBinary(char ch) {
    for (int i = 7; i >= 0; --i) { // Loop from 7 to 0 to get bits from MSB to LSB
        std::cout << ((ch >> i) & 1);
    }
}




//How to get element tableauMatrix[i][j]?
//  The byte contain the bit infor is  tableauMatrix[i*rowsize+j/8]
//  The index of the element in this byte is: j-8*(j/8)


//How to get the exact bit value of index k from a char A?
// int bit = (A >> k) & 1;


//If I have two bytes A and B, I want to get the XOR of the bit of A,B with index k?
//  ((A ^ B)>>k)&1;


//If I have two bytes A and B, I want to get the multplication of the bit of A,B with index k?
//  ((A & B)>>k)&1;


int main() {


  int num_qubit=5;

  int threadNum=2*num_qubit;


  // Every row is processed in a single thread, every thread is exactly one row of the tableau
  // Every tableau is process in a block, every block is exactly processed in one block
  int rowsize=((2*num_qubit+1)+7)/8; 
  int size=rowsize*(2*num_qubit);

  unsigned char* tableauMatrix =(unsigned char*) malloc(size);

  //Initialize the tableau
  for(int i=0;i<size;++i){
      tableauMatrix[i]=0;
  }

  int tmpindex;
  for(int k=0;k<2*num_qubit;k++){
       setTableauValue(tableauMatrix,rowsize, k, k,1);
  }


  unsigned char* cutableauMatrix;
  cudaMalloc(&cutableauMatrix,size);
  checkCudaError("cudaMalloc");

  cudaMemcpy(cutableauMatrix,tableauMatrix,size,cudaMemcpyHostToDevice); 
  checkCudaError("cudaMemcpy to device");  

  int threadsPerBlock = 2*num_qubit;
  int blocksPerGrid =1;

  H<<<blocksPerGrid, threadsPerBlock>>>(cutableauMatrix,1,num_qubit,rowsize,2*num_qubit);
  
  checkCudaError("Kernel launch");
  cudaDeviceSynchronize();
  checkCudaError("Kernel execution");

  cudaMemcpy(tableauMatrix,cutableauMatrix,size,cudaMemcpyDeviceToHost); 
  checkCudaError("cudaMemcpy to host");

  cudaFree(cutableauMatrix);
  checkCudaError("cudaFree");

  show_tableau_bit(tableauMatrix,num_qubit);  
  show_tableau_char(tableauMatrix,num_qubit);  

  return 0;
}
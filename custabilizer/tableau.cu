#include <cstdio> 
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <bitset>
#include <sstream>
#include <fstream>


#define _CONTROLNOT 0
#define _HADAMARD 1
#define _PHASE 2
#define _PAULIX 3
#define _PAULIY 4
#define _PAULIZ 5
#define _CONTROLZ 6



struct Instruction {
    int type;
    int target;
    int control;

    Instruction(const int& tp,const int& contr,const int& targ):type(tp),target(targ),control(contr){

    }

};


// Overload the << operator outside the struct without friend
std::ostream& operator<<(std::ostream& os, const  Instruction& inst);



__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}


__global__  void vector_add(float *A,float *B, float *C, int N) {
     int i=blockDim.x*blockIdx.x+threadIdx.x;
     if(i<N){
        C[i]=A[i]+B[i];
     }
}


#define getTableauElement(tableauMatrix,rowsize, row, col) (tableauMatrix[row*rowsize+col/8]&(0b10000000>>(col%8)))>>(7-(col%8))


#define setTableauValue(tableauMatrix,rowsize, row, col,value)       \
    do {                             \
        tableauMatrix[row*rowsize+col/8]=(value==1?(tableauMatrix[row*rowsize+col/8]|(0b00000000^(1<<(7-(col%8))))):(tableauMatrix[row*rowsize+col/8]&(0b11111111^(1<<(7-(col%8)))))); \
    } while (0)



__global__ void X(unsigned char* tableauMatrix,const size_t& target,const int& qubit_num,const int& rowsize,const int& N){

}


__global__ void Y(unsigned char* tableauMatrix,const size_t& target,const int& qubit_num,const int& rowsize,const int& N){

}


__global__ void Z(unsigned char* tableauMatrix,const size_t& target,const int& qubit_num,const int& rowsize,const int& N){

}


__global__ void P(unsigned char* tableauMatrix,size_t target,int qubit_num,int rowsize,int N){
    int row=threadIdx.x;
    if(row<N){
        int r=getTableauElement(tableauMatrix,rowsize, row, (2*qubit_num));
        int zi=getTableauElement(tableauMatrix,rowsize, row, target);
        int xi=getTableauElement(tableauMatrix,rowsize, row, (target+qubit_num));
        setTableauValue(tableauMatrix,rowsize, row, (2*qubit_num),((r+zi*xi)%2));
        setTableauValue(tableauMatrix,rowsize, row, target,((zi+xi)%2));
    }
}

__global__ void H(unsigned char* tableauMatrix,size_t target,int qubit_num,int rowsize,int N){
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


__global__ void CNOT(int *tableauMatrix,size_t control,size_t target,int qubit_num,int rowsize,int N){
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


__global__ void CZ(int *tableauMatrix,const size_t& control,const size_t& target,const int& qubit_num,const int& rowsize,int N){

}


// Function to print the binary representation of a char
void printBinary(char ch) {
    for (int i = 7; i >= 0; --i) { // Loop from 7 to 0 to get bits from MSB to LSB
        std::cout << ((ch >> i) & 1);
    }
}


void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}



void show_tableau_bit(const unsigned char* tableauMatrix,const int& num_qubit){
    int rowsize=((2*num_qubit+1)+7)/8; 
    int tmpindex;
    int showint;
    for(int row=0;row<2*num_qubit;row++){
          for(int col=0;col<2*num_qubit+1;col++){
                  //tmpindex=col-8*(col/8);
                  //showint=(tableauMatrix[row*rowsize+col/8]&(0b10000000>>tmpindex));
                  //showint=showint>>(7-tmpindex);
                  showint=getTableauElement(tableauMatrix,rowsize, row, col);
                  std::cout<<showint<<" ";
          }
          std::cout<<"\n";
    }
}



void show_tableau_char(const unsigned char* tableauMatrix,const int& num_qubit){
    int rowsize=((2*num_qubit+1)+7)/8; 
    int tmpindex;
    int zstabint;
    int xstabint;
    int phaseint;
    std::string tmpstr;
    for(int row=0;row<num_qubit;row++){
          tmpstr="";
          for(int col=0;col<num_qubit;col++){
                  zstabint=getTableauElement(tableauMatrix,rowsize, row, col);
                  xstabint=getTableauElement(tableauMatrix,rowsize, row, (col+num_qubit));
                  if((xstabint==0)&&(zstabint==0)){
                        tmpstr=tmpstr+"I";
                  }
                  else if((xstabint==1)&&(zstabint==0)){
                        tmpstr=tmpstr+"X";
                  }
                  else if((xstabint==0)&&(zstabint==1)){
                        tmpstr=tmpstr+"Z";
                  }
                  else{
                         tmpstr=tmpstr+"Y";
                  }
          }
          phaseint=getTableauElement(tableauMatrix,rowsize, row, (2*num_qubit));
          if(phaseint==1){
                tmpstr="-"+tmpstr;
          }
          std::cout<<tmpstr<<"\n";
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
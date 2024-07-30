#include <cstdio> 

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}


void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}


int main() {
    cuda_hello<<<1,1>>>(); 
    printf("Hello World from CPU!\n");
    return 0;
}
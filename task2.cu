#include <iostream> 
#include <cstddef>
#define BLOCK_SIZE 8
#define RADIUS 4
using namespace std;

__global__ void stencil_1d(double *in, double *out, size_t N) {
    __shared__ double temp[BLOCK_SIZE + 2 * RADIUS]; 
    int gindex = threadIdx.x + blockIdx.x * blockDim.x; 
    int lindex = threadIdx.x + RADIUS; 

    temp[lindex] = in[gindex]; 
    if (threadIdx.x < RADIUS) {
        if (gindex - RADIUS >= 0)
            temp[lindex - RADIUS] = in[gindex - RADIUS];
        if (gindex + BLOCK_SIZE < N)
            temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE]; 
    }

    __syncthreads();
    
    double result = 0;
    if (gindex >= RADIUS && gindex <= N - 1 - RADIUS){
        for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
            result += temp[lindex + offset];
        out[gindex - RADIUS] = result;
    }


}

int check_if_equal(double *a, double *b, int N){

    for (size_t i = 0; i != N; i++){
        if (a[i] != b[i]){
            return 1;
        }
    }
    
    return 0;
}

void print_array(string array_name, double *a, int N){
    cout << array_name + " = (";
    for (size_t i = 0; i != N; i++){
        cout << a[i];
        if (i != N-1) cout << ",";
        if (i > 20) {
            cout << "...";
            break;
        }
    }
    cout << ")" << endl;
}

void random_fill(double *array, int N, double random_lowest, double random_highest){

    const long max_rand = 1000000L;
    static double timep = 0.0;
    timep += 1.0;
    srandom(time(NULL) + timep);
    for (size_t i = 0; i != N; ++i){
    array[i] = random_lowest+(random_highest - random_lowest)*(random() % max_rand)/max_rand;
    }
}

void stencil_using_device(double *a, double *b, size_t N){
    double *d_a, *d_b; // device copies of a, b
    size_t size = N * sizeof(double);
    
    // Alloc space for device copies of a, b
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    stencil_1d<<<(N-1) / BLOCK_SIZE + 1,BLOCK_SIZE>>>(d_a, d_b, N);

    // Copy result back to host
    cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); 

}

void stencil_using_host(double *a, double *b, size_t N){
    double result;
    for (int i = RADIUS; i != N - RADIUS; i++){
        result = 0;
        for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
            result += a[i + offset];
        // Store the result
        b[i - RADIUS] = result;
    }
}

int main() {

    double *a, *b_host, *b_device; // host copies of a, b, c
    const size_t N = 12; // size of the arrays
    double random_lowest = 1.0; //lowest possible random double
    double random_highest = 10.0; //highest possible random double

    size_t size = N * sizeof(double);

    // Alloc space for host copies of a, b, c and setup input values
    a        = (double *)malloc(size);
    b_host   = (double *)malloc(size);
    b_device = (double *)malloc(size);

    random_fill(a, N, random_lowest, random_highest);

    print_array("a", a, N);

    //do c=a+b on device and copy the r
    stencil_using_device(a, b_device, N);
    stencil_using_host(a, b_host, N);

    cout << endl;
    print_array("b_device", b_device, N);
    print_array("b_host  ", b_host, N);
    cout << endl;

    //check if two results are equal
    if (check_if_equal(b_device, b_host, N)){
        cout << "Results from host and device are not equal!" << endl;
    }
    else {
        cout << "Results from host and device are equal!" << endl;
    }

    free(a); free(b_host); free(b_device);
    return 0;
}
#include <iostream> 
#include <cstddef>
using namespace std;

__global__ void add(double *a, double *b, double *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        c[index] = a[index] + b[index];
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
        if (i > 10) {
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

void add_using_device(double *a, double *b, double *c, size_t N, int M){
    double *d_a, *d_b, *d_c; // device copies of a, b, c
    size_t size = N * sizeof(double);
    
    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    add<<<(N-1) / M + 1,M>>>(d_a, d_b, d_c, N);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

}

void add_using_host(double *a, double *b, double *c, size_t N){

    for (size_t i = 0; i != N; i++){
        c[i] = a[i] + b[i];
    }
}

int main() {

    double *a, *b, *c_host, *c_device; // host copies of a, b, c
    const size_t N = 512; // size of the arrays
    const int THREADS_PER_BLOCK = 512;
    double random_lowest = 1.0; //lowest possible random double
    double random_highest = 10.0; //highest possible random double

    size_t size = N * sizeof(double);

    // Alloc space for host copies of a, b, c and setup input values
    a        = (double *)malloc(size);
    b        = (double *)malloc(size);
    c_host   = (double *)malloc(size);
    c_device = (double *)malloc(size);

    random_fill(a, N, random_lowest, random_highest);
    random_fill(b, N, random_lowest, random_highest);

    print_array("a", a, N);
    print_array("b", b, N);

    //do c=a+b on device and copy the r
    add_using_device(a, b, c_device, N, THREADS_PER_BLOCK);
    add_using_host(a, b, c_host, N);

    cout << endl;
    print_array("c_device", c_device, N);
    print_array("c_host  ", c_host, N);
    cout << endl;

    //check if two results are equal
    if (check_if_equal(c_device, c_host, N)){
        cout << "Results from host and device are not equal!" << endl;
    }
    else {
        cout << "Results from host and device are equal!" << endl;
    }


    free(a); free(b); free(c_host); free(c_device);
    return 0;
}
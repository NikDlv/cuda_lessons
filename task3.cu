#include <iostream> 
#include <cstddef>
#include <chrono>
#define BLOCK_SIZE 32
using namespace std;

__global__ void matmul_naive_1(double *a, double *b, double *c, size_t matrix_dim) {
  // compute position in C that this thread is responsible for
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < matrix_dim && y < matrix_dim) {
        float tmp = 0.0;
        for (int i = 0; i < matrix_dim; ++i) {
            tmp += a[x * matrix_dim + i] * b[i * matrix_dim + y];
        }
        c[x * matrix_dim + y] = tmp;
    }
}

__global__ void matmul_naive_2(double *a, double *b, double *c, size_t matrix_dim) {
  // compute position in C that this thread is responsible for
  const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < matrix_dim && y < matrix_dim) {
        float tmp = 0.0;
        for (int i = 0; i < matrix_dim; ++i) {
            tmp += a[y * matrix_dim + i] * b[i * matrix_dim + x];
        }
        c[y * matrix_dim + x] = tmp;
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

void print_matrix(string array_name, double *a, int N){
    //cout << array_name + " = ";
    for (size_t i = 0; i != N; i++){
        for (size_t j = 0; j != N; j++){

            cout << a[i * N + j] << " ";

        }

        cout << endl;
    }
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

void matmul_using_device(double *a, double *b, double *c, size_t matrix_dim,
void (*func)(double *a, double *b, double *c, size_t matrix_dim)){
    double *d_a, *d_b, *d_c; // device copies of a, b
    size_t array_dim = pow(matrix_dim,2); 
    size_t size = array_dim * sizeof(double);

    dim3 gridDim(matrix_dim / BLOCK_SIZE + 1, matrix_dim / BLOCK_SIZE + 1, 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
    
    // Alloc space for device copies of a, b
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    func<<<gridDim,blockDim>>>(d_a, d_b, d_c, matrix_dim);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); 

}

void matmul_simple_using_host(double *a, double *b, double *c, size_t N){

    for (size_t x = 0; x != N; x++){
        for (size_t y = 0; y != N; y++){
            float tmp = 0.0;
            for (int i = 0; i != N; ++i) {
                tmp += a[x * N + i] * b[i * N + y];
            }
            c[x * N + y] = tmp;
        }
    }
}

int main() {

    double *a, *b, *c_host, *c_device; // host copies of a, b, c
    const size_t matrix_dim = 1024*2; // size of the arrays
    const size_t array_dim = pow(matrix_dim, 2);
    double random_lowest = 1.0; //lowest possible random double
    double random_highest = 10.0; //highest possible random double
    std::chrono::steady_clock::time_point begin;
    std::chrono::steady_clock::time_point end;

    size_t size = array_dim * sizeof(double);

    // Alloc space for host copies of a, b, c and setup input values
    a        = (double *)malloc(size);
    b        = (double *)malloc(size);
    c_host   = (double *)malloc(size);
    c_device = (double *)malloc(size);

    random_fill(a, array_dim, random_lowest, random_highest);
    random_fill(b, array_dim, random_lowest, random_highest);
    
    // a[0] = 1.0;
    // a[1] = 2.0;
    // a[2] = 3.0;
    // a[3] = 4.0;

    // b[0] = 1.0;
    // b[1] = 2.0;
    // b[2] = 3.0;
    // b[3] = 4.0;

    // cout << "Matrix a:" << endl;
    // print_matrix("a", a, matrix_dim);
    // cout << "Matrix b:" << endl;
    // print_matrix("b", b, matrix_dim);
    begin = std::chrono::steady_clock::now();
    matmul_using_device(a, b, c_device, matrix_dim, &matmul_naive_1);
    end = std::chrono::steady_clock::now();
    std::cout << "matmul_naive_1 (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
    
    begin = std::chrono::steady_clock::now();
    matmul_using_device(a, b, c_device, matrix_dim, &matmul_naive_2);
    end = std::chrono::steady_clock::now();
    std::cout << "matmul_naive_2 (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;

    begin = std::chrono::steady_clock::now();
    matmul_simple_using_host(a, b, c_host, matrix_dim);
    end = std::chrono::steady_clock::now();
    std::cout << "host (sec) = " <<  (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;


    // cout << "Matrix c=a*b on host:" << endl;
    // print_matrix("c_host", c_host, matrix_dim);

    // cout << "Matrix c=a*b on device:" << endl;
    // print_matrix("c_device", c_device, matrix_dim);

    //check if two results are equal
    if (check_if_equal(c_device, c_host, array_dim)){
        cout << "Results from host and device are not equal!" << endl;
    }
    else {
        cout << "Results from host and device are equal!" << endl;
    }

    free(a); free(b); free(c_host); free(c_device);
    return 0;
}
#include <iostream> 
#include <cstddef>
#include "cublas_v2.h"
#include "assert.h"
using namespace std;

int check_if_equal(double *a, double *b, int N){

    for (size_t i = 0; i != N; i++){
        if (abs(a[i] - b[i]) > 1E-6){
            cout << a[i] <<" " <<  b[i] << " " << abs(a[i] - b[i]) << endl;
            return 1;
        }
    }
    
    return 0;
}

void print_matrix(string array_name, double *a, int N, int M){
    //cout << array_name + " = ";
    for (size_t i = 0; i != N; i++){
        for (size_t j = 0; j != M; j++){

            cout << a[j * N + i] << " ";

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

int main() {

    double *A, *B_long, *C_long, *C_long_async;
    double *A_device, *B_long_device, *C_long_device, *C_long_async_device;
    const size_t matrix_dim = 2048; // linear size of the matrix A  
    const size_t array_dim = pow(matrix_dim, 2); // number of elements in the matrix A
    const size_t N = 10; // how many times matrix A is in the long matrix
    double random_lowest = 1.0; //lowest possible random double
    double random_highest = 5.0; //highest possible random double
    double alpha = 1.0;
    double beta = 0.0;
    cublasStatus_t status;
    cudaError_t cudaerr;
    cublasHandle_t handle;
    cudaEvent_t start;
    cudaStream_t stream[N];
    float gpuTime = 0.0f;

    size_t size = array_dim * sizeof(double);

    // Alloc space for host copies of a, b, c and setup input values
    A              = (double *)malloc(size);
    B_long         = (double *)malloc(size*N);
    C_long         = (double *)malloc(size*N);
    C_long_async   = (double *)malloc(size*N);

    cudaMalloc((void **)&A_device, size);
    cudaMalloc((void **)&B_long_device, size*N);
    cudaMalloc((void **)&C_long_device, size*N);
    cudaMalloc((void **)&C_long_async_device, size*N);


    random_fill(A, array_dim, random_lowest, random_highest);
    random_fill(B_long, array_dim*N, random_lowest, random_highest);
    

    //cout << "Matrix A:" << endl;
    //print_matrix("A", A, matrix_dim, matrix_dim);
    //cout << "Matrix B_long:" << endl;
    //print_matrix("B_long", B_long, matrix_dim, matrix_dim*N);
    //cout << endl;

    //Serical
    cudaEventCreate(&start);
    cudaEventRecord(start, 0);
    cublasCreate(&handle);
    
    status = cublasSetVector(array_dim, sizeof(double), A, 1, A_device, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);
    status = cublasSetVector(array_dim*N, sizeof(double), B_long, 1, B_long_device, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);


    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_dim, matrix_dim*N, matrix_dim,
    &alpha, A_device, matrix_dim, B_long_device, matrix_dim, &beta, 
    C_long_device, matrix_dim);
    assert(status == CUBLAS_STATUS_SUCCESS);


    status = cublasGetVector(array_dim*N, sizeof(double), C_long_device, 1, C_long, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime ( &gpuTime, start, stop );
    printf("Serical Cublas: %.2f millseconds\n", gpuTime );
    //cout << "Matrix C_long:" << endl;
    //print_matrix("C_long", C_long, matrix_dim, matrix_dim*N);
    cout << endl;

    //Async
    cudaEventRecord(start, 0);
    cublasHandle_t handle_asyn;
    cublasCreate(&handle_asyn);
    status = cublasSetVector(array_dim, sizeof(double), A, 1, A_device, 1);
    assert(status == CUBLAS_STATUS_SUCCESS);


    for (int i = 0; i != N; i++){
        cudaerr = cudaStreamCreate(&stream[i]);
        assert(cudaerr == cudaSuccess);
    }
    for (int istream = 0; istream != N; istream++){

        status = cublasSetVectorAsync(array_dim, sizeof(double), B_long + array_dim * istream,
        1, B_long_device + array_dim * istream, 1, stream[istream]);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    for (int istream = 0; istream != N; istream++){
        status = cublasSetStream(handle_asyn, stream[istream]);
        assert(status == CUBLAS_STATUS_SUCCESS);
        
        status = cublasDgemm(handle_asyn, CUBLAS_OP_N, CUBLAS_OP_N, matrix_dim, matrix_dim, matrix_dim,
        &alpha, A_device, matrix_dim, B_long_device + array_dim * istream, matrix_dim, &beta, 
        C_long_async_device + array_dim * istream, matrix_dim);
        assert(status == CUBLAS_STATUS_SUCCESS);

    }

    for (int istream = 0; istream != N; istream++){

        status = cublasGetVectorAsync(array_dim, sizeof(double), C_long_async_device + array_dim * istream,
        1, C_long_async + array_dim * istream, 1, stream[istream]);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }

    for (int i = 0; i != N; i++){
        cudaerr = cudaStreamDestroy(stream[i]);
        assert(cudaerr == cudaSuccess);
    }

    cudaEventCreate(&stop);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime ( &gpuTime, start, stop );

    printf("Async Cublas: %.2f millseconds\n", gpuTime );
    //cout << "Matrix C_long_async:" << endl;
    //print_matrix("C_long_async", C_long_async, matrix_dim, matrix_dim*N);
    cout << endl;

    if (check_if_equal(C_long, C_long_async, array_dim*N)){
        cout << "Results from serical and async cublas are not equal!" << endl;
    }
    else {
        cout << "Results from serical and async cublas are equal!" << endl;
    }

    cudaFree(A_device); cudaFree(B_long_device); cudaFree(C_long_device);
    free(A); free(B_long); free(C_long); free(C_long_async);
    return 0;
}
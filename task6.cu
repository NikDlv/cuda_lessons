#include <iostream> 
#include <cstddef>
#include "cuda_runtime.h"
#include "cusolverDn.h"
#include <assert.h>
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


void random_fill_sym(double *a, int N, double random_lowest, double random_highest){

    const long max_rand = 1000000L;
    static double timep = 0.0;
    timep += 1.0;
    srandom(time(NULL) + timep);

    for (size_t i = 0; i != N; i++){
        for (size_t j = 0; j <= i; j++){
            a[j * N + i] = random_lowest+(random_highest - random_lowest)*(random() % max_rand)/max_rand;
            if (i != j) a[i * N + j] = a[j * N + i];
        }
    }
}

int main() {

    double *A, *eigenvalues;
    double *A_device, *eigenvalues_device;
    const size_t matrix_dim = 2; // linear size of the matrix A  
    const size_t array_dim = pow(matrix_dim, 2); // number of elements in the matrix A
    double random_lowest = 1.0; //lowest possible random double
    double random_highest = 5.0; //highest possible random double
    cusolverDnHandle_t cusolverH;
    cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
    cusolverDnParams_t dn_params;

    size_t size = array_dim * sizeof(double);
    size_t size_vec = matrix_dim * sizeof(double);

    A = (double *)malloc(size);
    eigenvalues = (double *)malloc(size_vec);


    cudaMalloc((void **)&A_device, size);
    cudaMalloc((void **)&eigenvalues_device, size);

    random_fill_sym(A, matrix_dim, random_lowest, random_highest);

    A[0] = 1.0;
    A[1] = 3.0;
    A[2] = 3.0;
    A[3] = 4.0;

    cudaMemcpy(A_device, A, size, cudaMemcpyHostToDevice);

    cout << "Matrix A:" << endl;
    print_matrix("A", A, matrix_dim, matrix_dim);
    cout << endl;

    
    
    cusolverDnCreateParams(&dn_params);
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    size_t workspaceDevice = 0;
    size_t workspaceHost = 0;
    cusolver_status = cusolverDnXsyevd_bufferSize(
        cusolverH, dn_params, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
        matrix_dim, CUDA_R_64F, A_device, matrix_dim, CUDA_R_64F, eigenvalues_device, CUDA_R_64F,
        &workspaceDevice, &workspaceHost);
    assert (cudaSuccess == cusolver_status);


    double *d_work, *h_work = 0;
    int *d_dev_info = 0;
    cudaMalloc((void**)&d_dev_info, sizeof(int));
    cudaMalloc((void**)&d_work, workspaceDevice);
    cudaMemset((void*)d_dev_info, 0, sizeof(int));
    cudaMemset((void*)d_work, 0, workspaceDevice);
    h_work = (double*)malloc(workspaceHost);
    
    
    cusolver_status = cusolverDnXsyevd(
        cusolverH, dn_params, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER,
        matrix_dim, CUDA_R_64F, A_device, matrix_dim, CUDA_R_64F,
        eigenvalues_device, CUDA_R_64F, d_work, workspaceDevice, h_work, workspaceHost,
        d_dev_info);
    assert (cudaSuccess == cusolver_status);

    cudaMemcpy(eigenvalues, eigenvalues_device, size_vec, cudaMemcpyDeviceToHost);

    print_array("eigenvalues", eigenvalues, matrix_dim);

    free(h_work);
    cudaFree(d_work);
    cudaFree(d_dev_info);
    cudaFree(eigenvalues_device);    
    cudaFree(A_device); 
    free(A); 
    return 0;
}
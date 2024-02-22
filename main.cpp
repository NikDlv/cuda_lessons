#include <iostream> 
#include <cstddef>
#include "utils.h"
#include "add.h"
using namespace std;

int main() {

    double *a, *b, *c_host, *c_device; // host copies of a, b, c
    const size_t N = 512; // size of the arrays
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
    add_using_device(a, b, c_device,N);
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


#include "utils.h"
#include <random>
#include <string>
#include <iostream>

using namespace std;

void random_fill(int *array, int N, int random_lowest, int random_highest){
    
    random_device dev;
    mt19937 rng(dev());
    uniform_int_distribution<mt19937::result_type> dist(random_lowest,random_highest);

    for (size_t i = 0; i != N; ++i){
        array[i] =  dist(rng);
    }
}

void random_fill(double *array, int N, double random_lowest, double random_highest){

    const long max_rand = 1000000L;
 
    srandom(time(NULL));
 
    for (size_t i = 0; i != N; ++i){
    array[i] = random_lowest+(random_highest - random_lowest)*(random() % max_rand)/max_rand;
    }
}

int check_if_equal(int *a, int *b, int N){

    for (size_t i = 0; i != N; i++){
        if (a[i] != b[i]){
            return 1;
        }
    }
    
    return 0;
}

int check_if_equal(double *a, double *b, int N){

    for (size_t i = 0; i != N; i++){
        if (a[i] != b[i]){
            return 1;
        }
    }
    
    return 0;
}

void print_array(string array_name, int *a, int N){
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
#include <iostream> 
void add(int *a, int *b, int *c);
void add_using_device(int *a, int *b, int *c, size_t N);
void add_using_host(int *a, int *b, int *c, size_t N);

void add(double *a, double *b, double *c);
void add_using_device(double *a, double *b, double *c, size_t N);
void add_using_host(double *a, double *b, double *c, size_t N);
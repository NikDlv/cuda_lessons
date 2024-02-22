#ifndef UTILS
#define UTILS
#include <string>
void random_fill(int *p, int N, int random_lowest, int random_highest);
void random_fill(double *p, int N, double random_lowest, double random_highest);
int check_if_equal(int *a, int *b, int N);
int check_if_equal(double *a, double *b, int N);
void print_array(std::string array_name, int *a, int N);
void print_array(std::string array_name, double *a, int N);
#endif  
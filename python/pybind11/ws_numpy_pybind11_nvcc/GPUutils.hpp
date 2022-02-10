#pragma once

double* cuda_vector_add(const double* data0, const double* data1, int num_element);

void cuda_vector_add(const double* data0, const double* data1, int num_element, double* ret);

void demo_vector_add(int num_element=5000);

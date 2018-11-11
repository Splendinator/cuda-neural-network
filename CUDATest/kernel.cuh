#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#include <cstdio>

#define E 2.718281828459


__global__ void randf(float *f, float *min, float *max);
__global__ void total(float *f, int size, float *bias);	//Totals up all numbers pointed to by f.
__global__ void mult(float *f, float *m);	//Multiplies f and m, storing result in dest.
__global__ void add(float *a, float *b);	//Adds a and b, storing result in a.
__global__ void multNodes(float *weights, float *values, const int numNodes);

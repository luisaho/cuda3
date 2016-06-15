/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/


#ifndef __SOLVER_H__
#define __SOLVER_H__

#include "def.h"

#ifdef __cplusplus
	extern "C" {
#endif

__device__ void matvec_csr(const int i, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y);
__device__ void matvec_ellpack(const int i, const int n, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y);
__device__ void matvec_spJDS(const int i, const int n, const floatType* data, const int* indices, const int* length, const int* permutation, const floatType* x, floatType* y);
__device__ void matvec_diagonal(const int i, const int n, const int num_dia, const int* offsets, const floatType* data, const floatType* x, floatType* y);
__device__ void unrolled_reduction(const int threadID, floatType *reduction_array_global_d);
void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc, const int* permutation);
    
//zusätzliche Funktionen für CUDA
__global__ void matvec_initialisierung(const int n, const int maxNNZ, const floatType* data_d, const int* indices_d, const int* length_d, const int* permutation_d, const floatType* b_d, const floatType* x_d, floatType *reduction_array_global_d, floatType* r_d, floatType* p_d, const int step);
__global__ void matvec_in_loop(const int n, const int maxNNZ, const floatType* data_d, const int* indices_d, const int* length_d, const int* permutation_d, floatType *reduction_array_global_d, const floatType* p_d, floatType* q_d, const int step);
__global__ void matvec_in_loop1(const int n, const int maxNNZ, const floatType* data_d, const int* indices_d, const int* length_d, const int* permutation_d, const floatType* p_d, floatType* q_d);
__global__ void matvec_in_loop2(const int n, floatType* reduction_array_global_d, const floatType* p_d, const floatType* q_d, const int step);
__global__ void mittelteil_ohne_x_update(const int n, const floatType alpha, floatType *reduction_array_global_d, floatType* r_d, const floatType* q_d, const int step);
__global__ void x_update(const int n, floatType* x_d, const floatType alpha, const floatType* p_d);
__global__ void p_update(const int n, const floatType beta, floatType* p_d, const floatType* r_d);

//getWTime, weil der Compiler das nicht in def.h findet...
double getWTime2();
void gpuWarmup2();

void printVector2(const floatType *x, int n);

#ifdef __cplusplus
	}
#endif


#endif

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


#ifndef __IO_H__
#define __IO_H__

#include "def.h"

#ifdef __cplusplus
extern "C" {
#endif
void parseMM(char *filename, int* n, int* nnz, int* maxNNZ, floatType** data, int** indices, int** length);
void printVector(const floatType *x, int n);
void printVectorInt(const int *x, const int n);
void printMatrix(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length);
void destroyMatrix(floatType* data, int* indices, int* length);
int toDIA(floatType* data, int* indices, int* length, const int n, floatType** dataDIA, int** offsetsDIA);
void toCSR(floatType* data, int* indices, int* length, const int N, const int nnz, floatType** dataCSR, int** indicesCSR, int** lengthCSR);
void toSPJDS(const floatType* data, const int* indices, const int* length, const int n, const int maxNNZ, floatType** dataSPJDS, int** indicesSPJDS, int** lengthSPJDS, int* permutation);


#ifdef __cplusplus
}
#endif

#endif

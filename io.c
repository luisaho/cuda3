/*****************************************************
 * CG Solver (HPC Software Lab)
 *
 * Parallel Programming Models for Applications in the 
 * Area of High-Performance Computation
 *====================================================
 * IT Center (ITC)
 * RWTH Aachen University, Germany
 * Author: Tim Cramer (cramer@itc.rwth-aachen.de)
 * 	   Fabian Schneider (f.schneider@itc.rwth-aachen.de)
 * Date: 2010 - 2015
 *****************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>

#ifndef _WIN32
# include <fcntl.h>
# include <unistd.h>
# include <sys/types.h>
# include <sys/stat.h>
# include <sys/mman.h>
#endif

#include "io.h"
#include "mmio.h"
#include "def.h"


void toSPJDS(const floatType* data, const int* indices, const int* length, const int n, const int maxNNZ, floatType** dataSPJDS, int** indicesSPJDS, int** lengthSPJDS, int* permutation){
	//Schritt 1: Zeilen sortieren (der Länge nach absteigend)
	printf("Starting to sort rows...");
	
	int i, j, k, l, current_index=0;
	floatType* dataNeu = malloc(sizeof(floatType) * n * maxNNZ);
	int* indicesNeu = malloc(sizeof(int) * n * maxNNZ);
	int* lengthNeu = malloc(sizeof(int) * n);
	
	//alle möglichen Zeilenlängen durchlaufen und jeweils alle Zeilen dieser Länge oben in den neuen Arrays einsortieren
	for(i = maxNNZ ; i > 0 ; i--){
		for(j = 0 ; j < n ; j++){
			if(length[j] == i){
				//Zeile j komplett verschieben
				for(k = 0 ; k < i ; k++){
					l = k * n;
					dataNeu[l + current_index] = data[l + j];
					indicesNeu[l + current_index] = indices[l + j];
				}
				lengthNeu[current_index] = length[j];
                permutation[current_index] = j;
				current_index++;
			}
		}
	}
	printf("  Done!\n");
	
	//Schritt 2: Padding und Slicing
	/* Idee:
	Matrix in Slices von 32 Zeilen aufteilen (und Vorbedingung: Zeilen nach Länge sortiert)
	alle Threads im Warp laufen genausoviele Spalten ab, wie der erste Thread im Warp zu tun hat (ggf auch Null-Elemente)
	Array zur Speicherung der Zeilenlänge jedes dieser Slices (Größe (int)(n/32)+1)
	Array zur Speicherung der Blockanfänge (Sum sliceLength[i]*32)
	*/
	printf("Padding and Slicing started...");
	
	int slicecount = (int)(n/WARPSIZE);
	if(n%WARPSIZE != 0) slicecount++;
	*lengthSPJDS = malloc(sizeof(int) * (2*slicecount + 1));
	int* slice_position = (int*)((*lengthSPJDS) + slicecount);
	slice_position[0] = 0;
	
	//Größe der neuen Arrays ermitteln
	for(i = 0 ; i < n ; i++){
		if(i%WARPSIZE == 0){
			//wir sind gerade am Anfang (erste Zeile) eines Slices
			(*lengthSPJDS)[i/WARPSIZE] = lengthNeu[i];
			slice_position[(i/WARPSIZE)+1] = slice_position[i/WARPSIZE] + WARPSIZE*lengthNeu[i];
		}
	}
	
	*dataSPJDS = malloc(sizeof(floatType)*slice_position[slicecount]);
	*indicesSPJDS = malloc(sizeof(int)*slice_position[slicecount]);
	
	//neue Arrays dürfen nur aus 0 bestehen!
	for(i = 0 ; i < slice_position[slicecount] ; i++){
		(*dataSPJDS)[i] = 0;
		(*indicesSPJDS)[i] = 0;
	}
	
	//konvertieren
	for(i = 0 ; i < n ; i++){
		current_index = i%WARPSIZE;
		for (j = 0; j < (*lengthSPJDS)[(int)(i/WARPSIZE)]; j++) {
			k = j * n + i;	//k-tes Element in Matrix ist an Zeile i und Spalte j (bzw indices[k])
			//current_index markiert die aktuelle Position in der neuen Matrix lokal (also innerhalb eines Slices)
			(*dataSPJDS)[slice_position[(int)(i/WARPSIZE)] + current_index] = dataNeu[k];
			(*indicesSPJDS)[slice_position[(int)(i/WARPSIZE)] + current_index] = indicesNeu[k];
			current_index += WARPSIZE;
		}
	}
	
	printf("  Done!\n");
	
	free(dataNeu);
	free(indicesNeu);
	free(lengthNeu);
}

void toCSR(floatType* data, int* indices, int* length, const int N, const int nnz, floatType** dataCSR, int** indicesCSR, int** lengthCSR){
	printf("Starting CSR Conversion...");
	int i,j,k;

	*dataCSR = (floatType*) malloc(sizeof(floatType) * nnz);
	*indicesCSR = (int*) malloc(sizeof(int) * nnz);
	*lengthCSR = (int*) malloc(sizeof(int) * (N+1));
	
	//Konvertierung
	int aktuellePosition=0;
	(*lengthCSR)[0] = 0;

	for (i = 0; i<N; i++) {
		(*lengthCSR)[i+1] = (*lengthCSR)[i]; 
		
		for (j = 0; j < length[i]; j++) {
			k = j * N + i;
			if(data[k] != 0){
				(*dataCSR)[aktuellePosition] = data[k];
				(*indicesCSR)[aktuellePosition] = indices[k];
				(*lengthCSR)[i+1]++;
				aktuellePosition++;
			}
		}
	}

	printf("  Done\n");
}

//Achtung: toDIA braucht EWIG (kann man sicher auch effizienter programmieren...), ist aber in jedem Fall nich brauchbar, weil Matrix nicht wirklich sehr "diagonal" ist:
//> 314440 verschiedene nicht-0-Diagonale (ca jede 10te Diagonale nicht 0) -> data wäre ca 315000*n*sizeof(floatType) byte groß! (> maxint)
int toDIA(floatType* data, int* indices, int* length, const int n, floatType** dataDIA, int** offsetsDIA){
	printf("Start DIA Conversion\n");
	int i,j,k,l,dia,current_index;

	/*Diagonalformat:
        Data-Matrix der Größe [#Zeilen] * [#nicht-Null-Diagonale]
            -> jede Spalte enthält eine Diagonale; Elemente, die eigentlich außerhalb der ursprüngl. Matrix lagen, sind hier 0, Spalten werden (wie in ELLPACK-R) hintereinander gespeichert
        offsets-Array der Größe [#nicht-Null-Diagonale]
    */

    //Anzahl Diagonale berechnen, bzw offsets-Array aufstellen
    int* temp_array;
    temp_array = (int*)malloc(n * sizeof(int));
    int temp_array_length = 0;
    
    for (i = 0; i < n; i++) {   //Zeilen
		for (j = 0; j < length[i]; j++) {   //Ellpack-R-Spalten
			k = j * n + i;
            if(data[k] != 0){
                dia = indices[k] - i; //aktuelle Diagonale (Spalte minus Zeile) zwischenspeichern
                //prüfe, ob aktuelle Diagonale bereits im temp_array vorhanden
                current_index = 0;
                while((current_index < temp_array_length) && (temp_array[current_index] < dia)){
                    current_index++;
                }
                if(current_index==temp_array_length){
                    //aktuelle Diagonale noch nicht im temp_array, hinten anhängen
                    temp_array[current_index] = dia;
                    temp_array_length++;
                }
                else if(temp_array[current_index] > dia){
                    //aktuelle Diagonale noch nicht im temp_array, an richtiger Stelle (current_index) einsetzen, Rest nach hinten schieben
                    for(l = temp_array_length ; l > current_index ; l--){
                        temp_array[l] = temp_array[l-1];
                    }
                    temp_array[current_index] = dia;
                    temp_array_length++;
                }
                //else: die Diagonale ist bereits im temp_array, also nichts tun und freuen
            }
		}
		if(i%100000==0) printf("i = %d | temp_array_length = %d\n",i,temp_array_length); 
	}
    *offsetsDIA = (int*) malloc(sizeof(int) * temp_array_length);
    for(i = 0 ; i < temp_array_length ; i++){
        (*offsetsDIA)[i] = temp_array[i];
    }
    free(temp_array);

    printf("DIA offsets-Array fertig");
	
    //Daten kopieren
    *dataDIA = (floatType*) malloc(sizeof(floatType) * temp_array_length * n);
    for (i = 0; i < n; i++) {
		for (j = 0; j < length[i]; j++) {
            k = j * n + i;
            if(data[k] != 0){
                dia = indices[k] - i;
                //aktueller Matrixeintrag in Diagonale dia , also erst suchen, an welcher Stelle im offsets-Array das ist
                current_index = 0;
                while(((*offsetsDIA)[current_index] != dia) && (current_index < temp_array_length)){
                    current_index++;
                }
                (*dataDIA)[current_index * n + i] = data[k];
            }
            
        }    
    }

	printf("DIA Conversion Done\n");
	return temp_array_length;
}

/* Parse the matrix market file "filename" and return
 * the matrix in ELLPACK-R format in A. */
void parseMM(char *filename, int* n, int* nnz, int* maxNNZ, floatType** data, int** indices, int** length){

	int M,N;
	int i,j;
	int *I, *J, *offset;
	floatType *V;
	FILE *fp;
	MM_typecode matcode;

	printf("Start matrix parse.\n");

	/* Try to open the file and return a error if not possible */
	if ((fp = fopen(filename, "r")) == NULL) {
		printf("ERROR: Cant open file!\n");
		exit(1);
	}

	/* Read the banner of the matrix market matrix. You should
	 * not care about the details of the file format at this 
	 * point. Exit in case of unsupported files. */
	if (mm_read_banner(fp, &matcode) != 0) {
		printf("ERROR: Could not process Matrix Market banner.\n");
		exit(1);
	}

	/* This is how one can screen matrix types if their application 
	 * only supports a subset of the Matrix Market data types.
	 * Please dont care about the details! */
	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
	    mm_is_sparse(matcode)) {
		printf("ERROR: Sorry, this application does not support ");
		printf("Market Market type: [%s]\n",
		    mm_typecode_to_str(matcode));
		exit(1);
	}

	/* Find out size of sparse matrix from the file */
	if (mm_read_mtx_crd_size(fp, &M, &N, nnz) != 0) {
		printf("ERROR: Could not read matrix size!\n");
		exit(1);
	}

	/* Exit for non square matrices. */
	if (N != M) {
		printf("ERROR: Naahhh. Come on, give me a NxN matrix!\n");
		exit(1);
	}

	printf("Start memory allocation.\n");

	/* if the matrix is stored in the symmetric format we will
	 * increase the number of nnz to store the upper and lower triangular */
	if (mm_is_symmetric(matcode)){

		/* store upper and lower triangular */
		(*nnz) = 2 * (*nnz) - N;
	}



	/* Allocate some of the memory for the ELLPACK-R matrix */
	*length = (int*) malloc(sizeof(int) * N);

	/* Check if the memory was allocated successfully */
	if (*length == NULL) {
		puts("Out of memory!");
		exit(1);
	}


	/* Set the number of non zeros (nnz) and the dimension (n)
	 * of the matrix */
	*n = N;

	/* Alocate the temporary  memory for matrix market matrix which
	 * has to be converted to the ELLPACK-R format */
	I = (int*)malloc(sizeof(int) * (*nnz));
	J = (int*)malloc(sizeof(int) * (*nnz));
	V = (floatType*)malloc(sizeof(floatType) * (*nnz));

	/* Check if the memory was allocated successfully */
	if (I == NULL || J == NULL || V == NULL) {
		puts("Out of memory!");
		exit(1);
	}

	/* Allocate and initialize some more temporay memory */
	if ((offset = (int*)malloc(sizeof(int) * (*nnz))) == NULL) {
		puts("Out of memory!");
		exit(1);
	}

	memset(offset, 0, (*nnz) * sizeof(int));

	printf("Read from file.\n");

	/* Start reading the file and store the values */
	
	
	for (i = 0; i < (*nnz); i++) {
		fscanf(fp, "%d %d %lg\n", &I[i], &J[i], &V[i]);

		//count double if entry is not on diag and in symmetric file format
		if (I[i] != J[i] && mm_is_symmetric(matcode)){
			((*length)[I[i]-1])++;
			i++;
			I[i] = J[i-1];
			J[i] = I[i-1];

			I[i-1]--;  //adjust from 1-based to 0-based
			J[i-1]--;

			V[i] = V[i-1];
		}


			//Adjust from 1-based to 0-based which means that in
			//the matrix market file format the first index is
			//always 1, but in C the first index is always 0.
		I[i]--; 
		J[i]--;
		
		// Count entries in one row
		((*length)[I[i]])++;
	}


	printf("Start converting from MM to ELLPACK-R.\n");

	/* Get maximum Number of NNZs per row for the ELLPACK-R format */
	*maxNNZ = 0;
	for (i = 0; i < N; i++) {
		if ((*length)[i] > (*maxNNZ)) {
			(*maxNNZ) = (*length)[i];
		}
	}

	//printf("maxNNZ = %d\n",*maxNNZ);
	
	/* Allocate the rest of the memory for the ELLPACK-R matrix */
	*data = (floatType*) malloc(sizeof(floatType) * N * (*maxNNZ));
	*indices = (int*) malloc(sizeof(int) * N * (*maxNNZ));

	/* Check if the memory was allocated successfully */
	if (*data == NULL || *indices == NULL) {
		puts("Out of memory!");
		exit(1);
	}

	/* Convert from MM to ELLPACK-R */

	for (j = 0; j < (*nnz); j++){
		i = I[j];

		/* Store data and indices in column-major order */
		(*data)[offset[i] * N + i] = V[j];
		(*indices)[offset[i] * N + i] = J[j];
		
		offset[i]++;
	}
	
	/* Insert 0's for padding in data and indices array */
	for (i = 0; i < N; i++) {
		for (j = (*length)[i]; j < (*maxNNZ); j++) {
			(*data)[j * N + i] = 0.0;
			(*indices)[j * N + i] = 0;
		}
	}
	
	printf("MM Parse done.\n");

	/* Clean up */
	free(offset);
	free(I);
	free(J);
	free(V);
	fclose(fp);
}

/* Free the complete memory of the matrix in ELLPACK-R format */
void destroyMatrix(floatType* data, int* indices, int* length) {
	free(data);
	free(indices);
	free(length);
}

/* Print out to std the first n elements of the vector x */
void printVector(const floatType *x, const int n) {
	int i;

	printf("(");
	for (i = 0; i < n; i++)
		printf("%d:%e' ", i, x[i]);
	printf(")\n");
}

void printVectorInt(const int *x, const int n) {
	int i;

	printf("(");
	for (i = 0; i < n; i++)
		printf("%d ", x[i]);
	printf(")\n");
}

/* Print out the whole ELLPACK-R matrix to std */
void printMatrix(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length) {
	int i, j, k;

	for (i = 0; i < n; i++) {
		if (i == 0) {
			printf("Row %d: [", 0);
		} 
		else {
			printf("]\nRow %d: [", i);
		}
		for (j = 0; j < length[i]; j++) {
			k = j * n + i;
			printf("%d:", indices[k]);
			printf("%f' ", data[k]);
		}
	}

	printf("]\n");
}

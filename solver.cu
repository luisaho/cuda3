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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _OPENACC
# include <openacc.h>
#endif

//#ifdef CUDA
#include <cuda.h>
//#endif

#include "solver.h"
#include "output.h"

//wegen getWTime, s. unten
#if defined(_WIN32) || defined(_WIN64)
# include <windows.h>
#else
# include <sys/time.h>
#endif

//globale Variablen
// __device__ floatType* data_d;
// __device__ int* indices_d;
// __device__ int* length_d;
int grid_size_reduction, grid_size_normal, step;

volatile __shared__ floatType reduction_array_shared[BLOCK_SIZE_REDUCTION];

//Shared-Memory Cache Tricks für spJDS: Zeilenlänge und Anfang aller Warps (im Speicher der Matrix) werden im SM gespeichert, sodass alle Warps eines Blocks nur noch aus dem SM lesen (statt global)
//__shared__ int warp_start[BLOCK_SIZE_REDUCTION / WARPSIZE];
//__shared__ int warp_length[BLOCK_SIZE_REDUCTION / WARPSIZE];


__inline__ __device__ void matvec_csr(const int i, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y){
      floatType sum = 0.0;
      int k;
	    for (k = length[i]; k < length[i+1]; k++) {
		    sum = sum + data[k]*x[indices[k]];
	    }
      y[i] = sum;
}

__inline__ __device__ void matvec_ellpackr(const int i, const int n, const floatType* data, const int* indices, const int* length, const floatType* x, floatType* y){
	floatType temp, sum = 0.0;
	int j, k;
	
	for (j = 0; j < length[i]; j++) {
		k = j * n + i;
		temp = data[k];
		if(temp != 0)
			sum += temp * x[indices[k]];
	}
    y[i] = sum;
}

__inline__ __device__ void matvec_spJDS(const int i, const int n, const floatType* data, const int* indices, const int* slice_length, const int* permutation, const floatType* x, floatType* y){
	floatType temp, sum = 0.0;
	int k, j = 0;
	const int local_index = i%WARPSIZE;					//Zeilenindex innerhalb eines Warps
	const int warp_index = (int)(i/WARPSIZE);			//Index des aktuellen Warps
	int warp_count = (int)(n/WARPSIZE);
	if(n%WARPSIZE != 0) warp_count++;					//hier ist bereits der letzte meistens unvollständige Warp enthalten
	const int warp_position = *(slice_length + warp_count + warp_index);	//Anfangsposition des aktuellen Warps im Matrixspeicher
	
	while(j < WARPSIZE * slice_length[warp_index]){
		k = warp_position + local_index + j;
		temp = data[k];
		if(temp != 0)
			sum += temp * x[indices[k]];
		j += WARPSIZE;
	}
    y[permutation[i]] = sum;
}
//bei der folgenden Version holen 2 Threads pro Warp jew. ein Element aus dem length/position array, sodass alle anderen threads nur noch aus dem Shared Memory lesen brauchen
/*__inline__ __device__ void matvec_spJDS(const int i, const int n, const floatType* data, const int* indices, const int* slice_length, const int* permutation, const floatType* x, floatType* y){
	floatType temp, sum = 0.0;
	int j, k;
	const int local_index = i%WARPSIZE;					//Zeilenindex innerhalb eines Warps
	const int warp_index_global = (int)(i/WARPSIZE);	//Index des aktuellen Warps
	const int warp_index_local = warp_index_global % (BLOCK_SIZE_REDUCTION / WARPSIZE);	//Index des aktuellen Warps innerhalb eines Blocks
	if(local_index == 1)
		warp_start[warp_index_local] = *(slice_length + ((int)(n/WARPSIZE) + 1) + warp_index_global);
	if(local_index == 2)
		warp_length[warp_index_local] = slice_length[warp_index_global];
	__syncthreads();
	const int warp_start_l = warp_start[warp_index_local];
	const int warp_length_l = warp_length[warp_index_local];
	
	for(j = 0 ; j < warp_length_l ; j++){
		k = warp_start_l + local_index + WARPSIZE*j;
		temp = fetch_floatType(data_tex, k);
		if(temp != 0)
			sum += temp * x[indices[k]];
	}
    y[permutation[i]] = sum;
}*/

__inline__ __device__ void matvec_diagonal(const int i, const int n, const int num_dia, const int* offsets, const floatType* data, const floatType* x, floatType* y){
    //Diagonalformat: data speichert alle Einträge der versch. Diagonalen in Spalten, offset speichert die Offsets zur Hauptdiagonalen (+ über, - unter Hauptdiagonalen)
    floatType sum = 0.0;
    int j, col;
    
    for(j = 0; j < num_dia ; j++){
        col = i + offsets[j];
        if(col >= 0 && col < n)
            sum += data[n * j + i] * x[col];
    }
    y[i] = sum;
}

__inline__ __device__ void unrolled_reduction(const int threadID, floatType* reduction_array_global_d){
    if(threadID < 512)  reduction_array_shared[threadID] += reduction_array_shared[threadID + 512];
    __syncthreads();
    if(threadID < 256)  reduction_array_shared[threadID] += reduction_array_shared[threadID + 256];
    __syncthreads();
    if(threadID < 128)  reduction_array_shared[threadID] += reduction_array_shared[threadID + 128];
    __syncthreads();
    if(threadID < 64)   reduction_array_shared[threadID] += reduction_array_shared[threadID + 64];
    __syncthreads();
    if(threadID < 32){
        reduction_array_shared[threadID] += reduction_array_shared[threadID + 32];
        reduction_array_shared[threadID] += reduction_array_shared[threadID + 16];
        reduction_array_shared[threadID] += reduction_array_shared[threadID + 8];
        reduction_array_shared[threadID] += reduction_array_shared[threadID + 4];
        reduction_array_shared[threadID] += reduction_array_shared[threadID + 2];
        reduction_array_shared[threadID] += reduction_array_shared[threadID + 1];
		//ein Thread pro Block schreibt in global Memory, restliche Reduction auf dem Host
		if(threadID == 0){
			reduction_array_global_d[blockIdx.x] = reduction_array_shared[0];
		}
    }
}
	
 //unrolled-by-two version
/*inline void matvec(const int j, const floatType* a, const int* colidx, const int* rowstr, const floatType* p, floatType* w){

    int iresidue,k,i;
	floatType sum1, sum2;
	i = rowstr[j];
    iresidue = (rowstr[j+1]-i) % 2;
    sum1 = 0.0;
    sum2 = 0.0;
        
    if (iresidue == 1){
        sum1 = sum1 + a[i]*p[colidx[i]];
    }
    
	for (k = i+iresidue; k <= rowstr[j+1]-2; k += 2) {
		sum1 = sum1 + a[k]   * p[colidx[k]];
		sum2 = sum2 + a[k+1] * p[colidx[k+1]];
	}
    w[j] = sum1 + sum2;
}
*/
    //unrolled-by-8 version
/*inline void matvec(const int j, const floatType* a, const int* colidx, const int* rowstr, const floatType* p, floatType* w){
    int iresidue,k,i;
    i = rowstr[j]; 
        
    iresidue = (rowstr[j+1]-i) % 8;
    floatType sum = 0.0;
        
    for (k = i; k <= i+iresidue-1; k++) {
        sum = sum +  a[k] * p[colidx[k]];
    }
    
    for (k = i+iresidue; k <= rowstr[j+1]-8; k += 8) {
        sum = sum + a[k  ] * p[colidx[k  ]]
                  + a[k+1] * p[colidx[k+1]]
                  + a[k+2] * p[colidx[k+2]]
                  + a[k+3] * p[colidx[k+3]]
                  + a[k+4] * p[colidx[k+4]]
                  + a[k+5] * p[colidx[k+5]]
                  + a[k+6] * p[colidx[k+6]]
                  + a[k+7] * p[colidx[k+7]];
    }
    w[j] = sum;
}
*/
	

/*
Paralleles Setup:

    die relevanten Schleifen laufen n Mal    
    für G3_circuit gilt: nnz=7660826 , n=1585478
    für GRID_SIZE und BLOCK_SIZE ist nach CUDA-Einführungsfolien (möglichst) zu erfüllen:
        -GRID_SIZE: mind. >2*SM bei 14-16 SMs ; max 8 Blocks/SM, also maximal 14*8=112
        -BLOCK_SIZE: maximal 1024 ; Teiler von 1536 (=max #Threads pro SM) ; Vielfaches von 32 (Warpsize) und 192<BLOCK_SIZE
    -> offensichtlich muss jeder Thread mehrere Schleifendurchläufe abarbeiten, damit das hinhaut (?)
    
    Cacheline: 128 byte
    floatType: 8 byte
	global constant memory limit: 65536

Schleifen:
    alter Zugriff (s. function2 unten, da noch exemplarisch auskommentiert, sonst überall gelöscht):
        jeder Thread bearbeitet LOOP_SIZE im Speicher aufeinanderfolgende Elemente
    neuer Zugriff:
        gleichzeitige Threads greifen nah beieinander zu (-> vermutlich gut wegen Cachelines und Warps!) ; jeder Thread springt nach jeder Schleifeniteration um GRID_SIZE * BLOCK_SIZE Elemente vor
    
Datenabgleich zwischen Host und Device:
    für rho und dot_pq regelmäßig
    x am Ende von cg wieder zum host bringen
    der Rest inkl. allen Vektoren bleibt auf dem device

Reduction:
    Bsp reduction von x:
	- jeder Thread zählt alle x seiner Schleife zusammen und speichert das in einem array im shared memory seines Blocks (x_array_shared)
	- das array wird immer wieder in zwei Teile aufgeilt und für jedes Element im vorderen Teil addiert ein Thread dieses Element mit einem anderen (im hinteren Teil) zusammen
	- das Array wird solange halbiert, bis nur noch ein Element da ist (da steht dann die Summe des ganzen Blocks drin)
	- jew. der erste Thread eines Block kopiert dieses Ergebnis an eine Stelle im globalen Array x_array_global_d (device memory) der Größe GRID_SIZE
	- der Host holt sich dieses Array und zählt alle GRID_SIZE ELemente zusammen
    -> http://cuda-programming.blogspot.in/2013/01/vector-dot-product-in-cuda-c.html
	
ELLPACK-R vs CSR/NASA:
	http://www.nvidia.com/object/nvidia_research_pub_001.html
	
Texture-Memory:
	cached Global Memory, sinnvoll bei read-only Zugriff
	Problem: wird langsamer, wenn zu viel Speicher gefetcht wird (Cache zu klein?)
		-> bisher beste Kombination: length und data fetchen, indices nicht

Compile:
	nicht IEEE-konform, dafür aber schneller, weil nicht so präzise: -ftz=true -prec-div=false -prec-sqrt=false
		
ISSUES:
    getWTime will nicht mitkompilieren... (undefined reference to 'getWTime') -> für's erste getWTime unten eingebaut...

TODO:
    - Pinned Memory für alle array-transfers auf Hostseite (http://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/)
		-> umgesetzt für die Reduktion, aber laut NVVP ist unser Memcpy-Anteil an der Laufzeit extrem gering, also erstmal unwichtig
	- CPU mehr arbeiten lassen, parallel mit GPU (zB bei Matvec Zeilen aufteilen auf CPU/GPU)
	Struct of Arrays vs Array of Structs, Matrixzugriffe auf indices und data nah beieinander?

Fragen:
    warum laufen die Threads innerhalb eines Warps in unrolled_reduction() nicht synchron?!?!
	Batchsystem misst??
*/
	
void cg(const int n, const int nnz, const int maxNNZ, const floatType* data, const int* indices, const int* length, const floatType* b, floatType* x, struct SolverConfig* sc, const int* permutation){
    grid_size_reduction = (int)(((int)(n/BLOCK_SIZE_REDUCTION))/16);	//=96
    grid_size_normal = (int)(n/BLOCK_SIZE_NORMAL)+1;
	//printf("GRID_SIZE_REDUCTION = %d\nGRID_SIZE_NORMAL = %d\n",grid_size_reduction, grid_size_normal);
    step = BLOCK_SIZE_REDUCTION * grid_size_reduction;

    floatType alpha, beta, rho, rho_old, dot_pq, bnrm2;
    rho = 0;
    dot_pq=0;
	int iter, current_stream;
 	double timeMatvec_s, timeMatvec = 0;
	int i;
	
	floatType *data_d;
	int *indices_d, *length_d;
    int* permutation_d;
	
	//Speicher so einstellen, dass L1 Cache groß ist (48kb) und SharedMemory klein (16kb)
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	
	double timeMemcpy = getWTime2();
    //die arrays zum Rechnen nur noch auf device behalten
    floatType *r_d, *p_d, *q_d;
    cudaMalloc(&p_d , n * sizeof(floatType));
    cudaMalloc(&q_d , n * sizeof(floatType));
    cudaMalloc(&r_d , n * sizeof(floatType));
	
	//Matrix zum device kopieren und vorbereiten
	if(MATRIXFORMAT == 0){			//ELLPACK-R
		cudaMalloc(&data_d , sizeof(floatType)*n*maxNNZ);
		cudaMalloc(&indices_d , sizeof(int)*n*maxNNZ);
		cudaMalloc(&length_d , sizeof(int)*n);
		cudaMemcpy(data_d, data, sizeof(floatType)*n*maxNNZ, cudaMemcpyHostToDevice);
		cudaMemcpy(indices_d, indices, sizeof(int)*n*maxNNZ, cudaMemcpyHostToDevice);
		cudaMemcpy(length_d, length, sizeof(int)*n, cudaMemcpyHostToDevice);
	}else if(MATRIXFORMAT == 1){	//CSR
		cudaMalloc(&data_d , sizeof(floatType)*nnz);
		cudaMalloc(&indices_d , sizeof(int)*nnz);
		cudaMalloc(&length_d , sizeof(int)*(n+1));
		cudaMemcpy(data_d, data, sizeof(floatType)*nnz, cudaMemcpyHostToDevice);
		cudaMemcpy(indices_d, indices, sizeof(int)*nnz, cudaMemcpyHostToDevice);
		cudaMemcpy(length_d, length, sizeof(int)*(n+1), cudaMemcpyHostToDevice);
	}else if(MATRIXFORMAT == 2){	//sliced pJDS
		int slicecount = (int)(n/WARPSIZE);
		if(n%WARPSIZE != 0) slicecount++;
		int* slice_position = (int*)(length + slicecount);
		cudaMalloc(&data_d , sizeof(floatType) * slice_position[slicecount]);
		cudaMalloc(&indices_d , sizeof(int) * slice_position[slicecount]);
		cudaMalloc(&length_d , sizeof(int) * (2*slicecount + 1));
        cudaMalloc(&permutation_d , sizeof(int)*n);
		cudaMemcpy(data_d, data, sizeof(floatType)*slice_position[slicecount], cudaMemcpyHostToDevice);
		cudaMemcpy(indices_d, indices, sizeof(int)*slice_position[slicecount], cudaMemcpyHostToDevice);
		cudaMemcpy(length_d, length, sizeof(int) * (2*slicecount + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(permutation_d, permutation, sizeof(int)*n, cudaMemcpyHostToDevice);
	}

    //noch x und b auf device vorbereiten
    floatType *x_d, *b_d;
    cudaMalloc(&x_d , n * sizeof(floatType));
    cudaMalloc(&b_d , n * sizeof(floatType));
    cudaMemcpy(x_d, x, sizeof(floatType) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(floatType) * n, cudaMemcpyHostToDevice);

    //Arrays zur reduction
	//Arrays, die oft hin und her kopiert werden, als pinned Memory -> schnellerer Zugriff	(in der Praxis nicht wirklich ein Unterschied...)
    floatType *reduction_array_global;
	cudaError_t status = cudaMallocHost((void**)&reduction_array_global, sizeof(floatType) * grid_size_reduction);
	if (status != cudaSuccess) printf("Error allocating pinned host memory\n");
    floatType* reduction_array_global_d;
	cudaMalloc(&reduction_array_global_d , sizeof(floatType) * grid_size_reduction);
	
	//printf("cudaMemcpy und cudaMalloc - Zeit: %f\n", getWTime2()-timeMemcpy);
	
	//Streams , ein bisschen tricksen, damit in jeder Iteration jeder Stream die "ID" wechselt (sinnvoll, damit vom letzten Stream der Schleife zum ersten kein synchronize nötig ist)
	cudaStream_t streams[2];
	cudaStreamCreate(&streams[0]);
	cudaStreamCreate(&streams[1]);

    //erster Kernelaufruf
    matvec_initialisierung<<<grid_size_reduction , BLOCK_SIZE_REDUCTION>>>(n, maxNNZ, data_d, indices_d, length_d, permutation_d, b_d, x_d, reduction_array_global_d, r_d, p_d, step);

    //residual initialisieren
	sc->residual = sc->tolerance*2;

    //reduction von rho
    cudaMemcpy(reduction_array_global , reduction_array_global_d , grid_size_reduction * sizeof(floatType) , cudaMemcpyDeviceToHost);
    for(i = 0 ; i < grid_size_reduction ; i++){
        rho += reduction_array_global[i];
    }
    
	printf("rho_0=%e\n", rho);

    //rho ist hier das Vektorprodukt <r,r>, also gilt ||r||_2 = sqrt(<r,r>)
	bnrm2 = 1.0 / sqrt(rho);

    //eigentliche Schleife beginnt hier
    for(iter = 0; iter < sc->maxIter && sc->residual > sc->tolerance; iter++){
		current_stream = iter%2;
		
        //zweiter Kernelaufruf
        if(MATRIXFORMAT == 0 || MATRIXFORMAT == 1)
			matvec_in_loop<<<grid_size_reduction , BLOCK_SIZE_REDUCTION , 0 , streams[current_stream]>>>(n, maxNNZ, data_d, indices_d, length_d, permutation_d, reduction_array_global_d, p_d, q_d, step);
		else if(MATRIXFORMAT == 2){
			//Aufsplitten ist nötig, damit alle Threads nachdem durchlaufen von matvec global synchronisiert sind (wegen permutatierten/normalen Zugriffen auf q_d)
			matvec_in_loop1<<<grid_size_normal , BLOCK_SIZE_NORMAL , 0 , streams[current_stream]>>>(n, maxNNZ, data_d, indices_d, length_d, permutation_d, p_d, q_d);
			matvec_in_loop2<<<grid_size_reduction , BLOCK_SIZE_REDUCTION , 0 , streams[current_stream]>>>(n, reduction_array_global_d, p_d, q_d, step);
		}
		timeMatvec_s = getWTime2();
		
        rho_old = rho;
		
        //reduction von dot_pq
        cudaMemcpyAsync(reduction_array_global , reduction_array_global_d , grid_size_reduction * sizeof(floatType) , cudaMemcpyDeviceToHost, streams[current_stream]);
        cudaStreamSynchronize(streams[current_stream]);
		
		for(i = 0 ; i < grid_size_reduction ; i++){
            dot_pq += reduction_array_global[i];
        }

		timeMatvec += getWTime2() - timeMatvec_s;

		/* alpha     = rho(k) / dot_pq */
		alpha = (rho / dot_pq);
		
		dot_pq = 0;
        rho = 0;
		
	    //zwei Kernelaufrufe hintereinander, damit der zweite während der reduction des ersten laufen kann
		mittelteil_ohne_x_update<<<grid_size_reduction , BLOCK_SIZE_REDUCTION , 0 , streams[current_stream]>>>(n, alpha, reduction_array_global_d, r_d, q_d, step);
		
		//printf("res_%d=%e\n", iter, sc->residual);
		
		//reduction von rho
		cudaMemcpyAsync(reduction_array_global , reduction_array_global_d , grid_size_reduction * sizeof(floatType) , cudaMemcpyDeviceToHost , streams[current_stream]);
		
		x_update<<<grid_size_normal , BLOCK_SIZE_NORMAL , 0 , streams[(current_stream+1)%2]>>>(n, x_d, alpha, p_d);
		cudaStreamSynchronize(streams[current_stream]);
        for(i = 0 ; i < grid_size_reduction ; i++){
            rho += reduction_array_global[i];
        }
		
		/* beta      = rho(k+1) / rho(k) */
		beta = rho / rho_old;
		
        //vierter Kernelaufruf
        p_update<<<grid_size_normal , BLOCK_SIZE_NORMAL , 0 , streams[(current_stream+1)%2]>>>(n, beta, p_d, r_d);

        /* Normalize the residual with initial one */
		sc->residual = sqrt(rho) * bnrm2;
    }
    
    //x zurückholen
    cudaMemcpy(x, x_d, sizeof(floatType) * n, cudaMemcpyDeviceToHost);
    
    sc->iter = iter;
	sc->timeMatvec = timeMatvec;
	
	//Streams schließen
	cudaStreamDestroy(streams[0]);
	cudaStreamDestroy(streams[1]);

    //Speicher auf Device freigeben
    cudaFree(r_d);
	cudaFree(p_d);
	cudaFree(q_d);
	cudaFree(x_d);
    cudaFree(b_d);
	cudaFree(data_d);
	cudaFree(indices_d);
	cudaFree(length_d);
    cudaFree(reduction_array_global_d);
}

//Kernels
__global__ void matvec_initialisierung(const int n, const int maxNNZ, const floatType* data_d, const int* indices_d, const int* length_d, const int* permutation_d, const floatType* b_d, const floatType* x_d, floatType* reduction_array_global_d, floatType* r_d, floatType* p_d, const int step){
    const int threadID = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadID;
    floatType temp, rho_local = 0.0;
    
    while(id < n){
        /* r(0)    = Ax(0); */
        /*if(MATRIXFORMAT == 0)		//ELLPACK-R
			matvec_ellpackr(id, n, data_d, indices_d, length_d, x_d, r_d);
		else if(MATRIXFORMAT == 1)	//CSR
			matvec_csr(id, data_d, indices_d, length_d, x_d, r_d);
		else if(MATRIXFORMAT == 2)	//sliced pJDS
			matvec_spJDS(id, n, data_d, indices_d, length_d, permutation_d, x_d, r_d);
		*/
		
        /* r(0)    = b - Ax(0) */
        temp = b_d[id];// - r_d[id]; <- unnötig, weil x0 = 0...0, eigentlich ist der gesamte erste matvec-Aufruf unnötig
        r_d[id] = temp;
        /* p(0)    = r(0) */
        p_d[id] = temp;
        /* rho(0)    =  <r(0),r(0)> */
        rho_local += temp * temp;
		
		id += step;
    }
	
    reduction_array_shared[threadID] = rho_local;
    __syncthreads();
    unrolled_reduction(threadID, reduction_array_global_d);
}

__global__ void matvec_in_loop(const int n, const int maxNNZ, const floatType* data_d, const int* indices_d, const int* length_d, const int* permutation_d, floatType* reduction_array_global_d, const floatType* p_d, floatType* q_d, const int step){
    const int threadID = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadID;
    floatType dot_pq_local = 0.0;
	
    while(id < n){
		if(MATRIXFORMAT == 0)		//ELLPACK-R
			matvec_ellpackr(id, n, data_d, indices_d, length_d, p_d, q_d);
		else if(MATRIXFORMAT == 1)	//CSR
			matvec_csr(id, data_d, indices_d, length_d, p_d, q_d);
		else if(MATRIXFORMAT == 2)	//sliced pJDS
			matvec_spJDS(id, n, data_d, indices_d, length_d, permutation_d, p_d, q_d);
			
        dot_pq_local += p_d[id] * q_d[id];
		id += step;
    }

    reduction_array_shared[threadID] = dot_pq_local;
    __syncthreads();
    unrolled_reduction(threadID, reduction_array_global_d);
}

__global__ void matvec_in_loop1(const int n, const int maxNNZ, const floatType* data_d, const int* indices_d, const int* length_d, const int* permutation_d, const floatType* p_d, floatType* q_d){
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < n){
		if(MATRIXFORMAT == 0)		//ELLPACK-R
			matvec_ellpackr(id, n, data_d, indices_d, length_d, p_d, q_d);
		else if(MATRIXFORMAT == 1)	//CSR
			matvec_csr(id, data_d, indices_d, length_d, p_d, q_d);
		else if(MATRIXFORMAT == 2)	//sliced pJDS
			matvec_spJDS(id, n, data_d, indices_d, length_d, permutation_d, p_d, q_d);
	}
}
__global__ void matvec_in_loop2(const int n, floatType* reduction_array_global_d, const floatType* p_d, const floatType* q_d, const int step){
	const int threadID = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadID;
    floatType dot_pq_local = 0.0;

	while(id < n){
		dot_pq_local += p_d[id] * q_d[id];
		id += step;
	}
	
	reduction_array_shared[threadID] = dot_pq_local;
    __syncthreads();
    unrolled_reduction(threadID, reduction_array_global_d);
}

__global__ void mittelteil_ohne_x_update(const int n, const floatType alpha, floatType* reduction_array_global_d, floatType* r_d, const floatType* q_d, const int step){
    const int threadID = threadIdx.x;
    int id = blockIdx.x * blockDim.x + threadID;
    floatType temp, rho_local = 0.0;
    
    while(id < n){
		temp = r_d[id] - alpha*q_d[id];
		r_d[id] = temp;
		rho_local += temp * temp;
		
		id += step;
    }
	
    reduction_array_shared[threadID] = rho_local;
    __syncthreads();
    unrolled_reduction(threadID, reduction_array_global_d);
}

__global__ void x_update(const int n, floatType* x_d, const floatType alpha, const floatType* p_d){
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
		x_d[id] = x_d[id] + alpha*p_d[id];
    }
}

__global__ void p_update(const int n, const floatType beta, floatType* p_d, const floatType* r_d){
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n){
		p_d[id] = r_d[id] + beta * p_d[id];
    }
}


//aus irgendeinem Grund findet der Compiler getWTime() aus def.h nicht?!?! deshalb hier einfach reingeklatscht
/* Use is time function to get the real time */
double getWTime2() {
#if defined(_WIN32) || defined(_WIN64)
# define Li2Double(x) ((double)((x).HighPart) * 4.294967296E9 + \
                      (double)((x).LowPart))
	LARGE_INTEGER time, freq;
	double dtime, dfreq, res;

	if (QueryPerformanceCounter(&time) == 0) {
		DWORD err = GetLastError();
		LPVOID buf;
		FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
		    FORMAT_MESSAGE_FROM_SYSTEM, NULL, err,
		    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buf,
		    0, NULL);
		printf("QueryPerformanceCounter() failed with error %d: %s\n",
		    err, buf);
		exit(1);
	}

	if (QueryPerformanceFrequency(&freq) == 0)
	{
		DWORD err = GetLastError();
		LPVOID buf;
		FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER |
		    FORMAT_MESSAGE_FROM_SYSTEM, NULL, err,
		    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR) &buf,
		    0, NULL);
		printf("QueryPerformanceFrequency() failed with error %d: %s\n",
		    err, buf);
		exit(1);
	}

	dtime = Li2Double(time);
	dfreq = Li2Double(freq);
	res = dtime / dfreq;

	return res;
#else
	struct timeval tv;
	gettimeofday(&tv, (struct timezone*)0);
	return ((double)tv.tv_sec + (double)tv.tv_usec / 1000000.0 );
#endif
}

void gpuWarmup2(){
#ifdef _OPENACC
  acc_init(acc_device_nvidia);
#endif

#ifdef CUDA
	floatType* a;
	cudaMalloc((void**)&a, 1000);
  cudaFree(a);
#endif
}
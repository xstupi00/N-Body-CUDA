/**
 * @File nbody.h
 *
 * Header file of your implementation to the N-Body problem
 *
 * Parallel Computations on GPU (PCG 2020)
 * Assignment no. 1 (cuda)
 * Login: xstupi00
 */

#ifndef __NBODY_H__
#define __NBODY_H__

#include <cstdlib>
#include <cstdio>
#include "h5Helper.h"

/* Gravitation constant */
constexpr float G =  6.67384e-11f;
constexpr float COLLISION_DISTANCE = 0.01f;


/**
 * Particles data structure
 */
typedef struct
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //                 FILL IN: Particle data structure optimal for the use on GPU (step 0)                             //
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  float* pos_x;
  float* pos_y;
  float* pos_z;
  float* vel_x;
  float* vel_y;
  float* vel_z;
  float* weight;
} t_particles;

/**
 * CUDA kernel to calculate gravitation and collision velocity and update particles.
 * @param p_in    - input particles
 * @param p_out   - output particles
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_velocity(t_particles p, int N, float dt);

/*
 * CUDA device function to perform the WARP-synchronous programming within the reduction.
 * @param pos_x     - positions x from particles data
 * @param pos_y     - positions y from particles data
 * @param pos_z     - positions z from particles data
 * @param weights   - weights from particles data
 * @param tid       - computed thread index
 * @param blockSize - size of the block
 */
inline __device__ void warpReduce(
        volatile float* pos_x, volatile float* pos_y, volatile float* pos_z,
        volatile float* weights, unsigned int tid, unsigned int blockSize
) {
    // Reading and writing to the shared memory in the SIMD takes place synchronously inside the warp.
    // We can cancel __syncthreads() and expand the last six iterations.

    pos_x[tid] += pos_x[tid + 32];
    pos_y[tid] += pos_y[tid + 32];
    pos_z[tid] += pos_z[tid + 32];
    weights[tid] += weights[tid + 32];

    pos_x[tid] += pos_x[tid + 16];
    pos_y[tid] += pos_y[tid + 16];
    pos_z[tid] += pos_z[tid + 16];
    weights[tid] += weights[tid + 16];

    pos_x[tid] += pos_x[tid + 8];
    pos_y[tid] += pos_y[tid + 8];
    pos_z[tid] += pos_z[tid + 8];
    weights[tid] += weights[tid + 8];

    pos_x[tid] += pos_x[tid + 4];
    pos_y[tid] += pos_y[tid + 4];
    pos_z[tid] += pos_z[tid + 4];
    weights[tid] += weights[tid + 4];

    pos_x[tid] += pos_x[tid + 2];
    pos_y[tid] += pos_y[tid + 2];
    pos_z[tid] += pos_z[tid + 2];
    weights[tid] += weights[tid + 2];

    pos_x[tid] += pos_x[tid + 1];
    pos_y[tid] += pos_y[tid + 1];
    pos_z[tid] += pos_z[tid + 1];
    weights[tid] += weights[tid + 1];

}

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param comX    - pointer to a center of mass position in X
 * @param comY    - pointer to a center of mass position in Y
 * @param comZ    - pointer to a center of mass position in Z
 * @param comW    - pointer to a center of mass weight
 * @param lock    - pointer to a user-implemented lock
 * @param N       - Number of particles
 */
template <bool nIsPow2>
__global__ void centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N)
{
    // Declares dynamic allocation of the shared memory
    extern __shared__ float shared_data[];
    unsigned int blockSize = blockDim.x;    // block size
    // Obtains the pointers to the relevant parts of the shared memory for individual items of particles data
    float* pos_x = shared_data;
    float* pos_y = &shared_data[blockSize];
    float* pos_z = &shared_data[blockSize * 2];
    float* weights = &shared_data[blockSize * 3];

    unsigned int tid = threadIdx.x;         // thread index
    // Computes the global index of thread within the grid
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    // Computes the reduction size of the whole program grid
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    // Clears the individual items of particles data by each thread before the starting of reduction
    pos_x[tid] = 0.0f;
    pos_y[tid] = 0.0f;
    pos_z[tid] = 0.0f;
    weights[tid] = 0.0f;

    // Each thread have to covered the relevant number of particles according to the program grid size
    while (i < N) {
        // Loads the weights of both particles to reduce the duplicate readings from the memory
        float weight_i = p.weight[i];
        float weight_j = p.weight[i + blockSize];
        // Computes the addition over the relevant particles data according to the requirements of reduction
        pos_x[tid] += (nIsPow2 || i + blockSize < N) ?
                      (p.pos_x[i] * weight_i) + (p.pos_x[i + blockSize] * weight_j) : (p.pos_x[i] * weight_i);
        pos_y[tid] += (nIsPow2 || i + blockSize < N) ?
                      (p.pos_y[i] * weight_i) + (p.pos_y[i + blockSize] * weight_j) : (p.pos_y[i] * weight_i);
        pos_z[tid] += (nIsPow2 || i + blockSize < N) ?
                      (p.pos_z[i] * weight_i) + (p.pos_z[i + blockSize] * weight_j) : (p.pos_z[i] * weight_i);
        weights[tid] += (nIsPow2 || i + blockSize < N) ? weight_i + weight_j : weight_i;
        // Shift the index for covering the next particle according to the program grid size
        i += gridSize;
    }
    // Synchronizes all threads within the block until they computed the own part of the reduction
    __syncthreads();

    // Iterates over the partial results of performed reduction
    for(unsigned int stride = blockSize / 2; stride > 32; stride >>= 1) {
        // Computes the addition over the relevant particles data according to the requirements of reduction
        pos_x[tid] += (tid < stride) ? pos_x[tid + stride] : 0.0f;
        pos_y[tid] += (tid < stride) ? pos_y[tid + stride] : 0.0f;
        pos_z[tid] += (tid < stride) ? pos_z[tid + stride] : 0.0f;
        weights[tid] += (tid < stride) ? weights[tid + stride] : 0.0f;
        // Synchronizes all threads within the block until they computed the own part of the computation
        __syncthreads();
    }

    // Perform the warp reduce when the number of operations are executed only by one warp
    if (tid < 32) { // expanded the last six iterations
        warpReduce(pos_x, pos_y, pos_z, weights, tid, blockSize);
    }

    // Thread with index 0 within the block writes the result of the reduction to the GPU memory after the computation
    if (tid == 0) {
        // Ensures the mutual exclusion for reducing the result to the global memory between the different blocks
        while (0 != atomicCAS(lock, 0, 1)) {}
        // Writes the individual results of the reduction to the final results into global memory GPU
        *comX += pos_x[tid];
        *comY += pos_y[tid];
        *comZ += pos_z[tid];
        *comW += weights[tid];
        // Returns the lock to write to the shared global memory GPU
        atomicExch(lock, 0);
    }

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------


/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassCPU(MemDesc& memDesc);

/**
 * Check whether the given numner is the power of number two or it is not.
 * @param x - Number to check whether it is the power of two
 */
bool ispow2(int x);

#endif /* __NBODY_H__ */

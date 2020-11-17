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
__global__ void calculate_velocity(t_particles p_in, t_particles p_out, int N, float dt);

template <unsigned int blockSize>
__device__ void
warpReduce(
        volatile float* pos_x, volatile float* pos_y, volatile float* pos_z, volatile float* weights, unsigned int tid
) {
    if (blockSize >= 64) {
        pos_x[tid] += pos_x[tid + 32];
        pos_y[tid] += pos_y[tid + 32];
        pos_z[tid] += pos_z[tid + 32];
        weights[tid] += weights[tid + 32];
    }
//    __syncwarp();

    if (blockSize >= 32) {
        pos_x[tid] += pos_x[tid + 16];
        pos_y[tid] += pos_y[tid + 16];
        pos_z[tid] += pos_z[tid + 16];
        weights[tid] += weights[tid + 16];
    }
//    __syncwarp();

    if (blockSize >= 16) {
        pos_x[tid] += pos_x[tid + 8];
        pos_y[tid] += pos_y[tid + 8];
        pos_z[tid] += pos_z[tid + 8];
        weights[tid] += weights[tid + 8];
    }
//    __syncwarp();

    if (blockSize >= 8) {
        pos_x[tid] += pos_x[tid + 4];
        pos_y[tid] += pos_y[tid + 4];
        pos_z[tid] += pos_z[tid + 4];
        weights[tid] += weights[tid + 4];
    }
//    __syncwarp();

    if (blockSize >= 4) {
        pos_x[tid] += pos_x[tid + 2];
        pos_y[tid] += pos_y[tid + 2];
        pos_z[tid] += pos_z[tid + 2];
        weights[tid] += weights[tid + 2];
    }
//    __syncwarp();

    if (blockSize >= 2) {
        pos_x[tid] += pos_x[tid + 1];
        pos_y[tid] += pos_y[tid + 1];
        pos_z[tid] += pos_z[tid + 1];
        weights[tid] += weights[tid + 1];
    }
//    __syncwarp();
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
template <unsigned int blockSize, bool nIsPow2>
__global__ void
centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N)
{
    extern __shared__ float shared_data[];
    float* pos_x = shared_data;
    float* pos_y = &shared_data[blockSize];
    float* pos_z = &shared_data[blockSize * 2];
    float* weights = &shared_data[blockSize * 3];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize * 2 * gridDim.x;

    pos_x[tid] = 0.0f;
    pos_y[tid] = 0.0f;
    pos_z[tid] = 0.0f;
    weights[tid] = 0.0f;

    while (i < N) {
        float weight_i = p.weight[i];
        float weight_j = p.weight[i + blockSize];
        pos_x[tid] += (nIsPow2 || i + blockSize < N) ?
            (p.pos_x[i] * weight_i) + (p.pos_x[i + blockSize] * weight_j) : (p.pos_x[i] * weight_i);
        pos_y[tid] += (nIsPow2 || i + blockSize < N) ?
            (p.pos_y[i] * weight_i) + (p.pos_y[i + blockSize] * weight_j) : (p.pos_y[i] * weight_i);
        pos_z[tid] += (nIsPow2 || i + blockSize < N) ?
            (p.pos_z[i] * weight_i) + (p.pos_z[i + blockSize] * weight_j) : (p.pos_z[i] * weight_i);
        weights[tid] += (nIsPow2 || i + blockSize < N) ? weight_i + weight_j : weight_i;
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 1024 && tid < 512) {
        pos_x[tid] += pos_x[tid + 512];
        pos_y[tid] += pos_y[tid + 512];
        pos_z[tid] += pos_z[tid + 512];
        weights[tid] += weights[tid + 512];
    }
    __syncthreads();

    if (blockSize >= 512 && tid < 256) {
        pos_x[tid] += pos_x[tid + 256];
        pos_y[tid] += pos_y[tid + 256];
        pos_z[tid] += pos_z[tid + 256];
        weights[tid] += weights[tid + 256];
    }
    __syncthreads();

    if (blockSize >= 256 && tid < 128) {
        pos_x[tid] += pos_x[tid + 128];
        pos_y[tid] += pos_y[tid + 128];
        pos_z[tid] += pos_z[tid + 128];
        weights[tid] += weights[tid + 128];
    }
    __syncthreads();

    if (blockSize >= 128 && tid < 64) {
        pos_x[tid] += pos_x[tid + 64];
        pos_y[tid] += pos_y[tid + 64];
        pos_z[tid] += pos_z[tid + 64];
        weights[tid] += weights[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        warpReduce<blockSize>(pos_x, pos_y, pos_z, weights, tid);
    }

    if (tid == 0) {
        while (0 != atomicCAS(lock, 0, 1)) {}
        *comX += pos_x[tid];
        *comY += pos_y[tid];
        *comZ += pos_z[tid];
        *comW += weights[tid];
        atomicExch(lock, 0);
    }

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

void compute_gpu_center_of_mass(
        t_particles p, float4* com, int* lock, const int N, const int red_grid,
        const int red_thr, size_t shm_mem, cudaStream_t* m_stream
);

/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassCPU(MemDesc& memDesc);

#endif /* __NBODY_H__ */

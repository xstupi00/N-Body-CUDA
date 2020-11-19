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
/* Initialize the warp size for computation of the reduction */
constexpr int WARP_SIZE = 32;


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
extern __global__
void centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N);


/**
 * CUDA kernel function to perform reduction across the entire block
 * @param partial_sums  - partial sums within individual threads in the blocks
 */
__inline__ __device__ float4 block_reduce_sum(float4 partial_sums);


/**
 * CUDA kernel function to perform shuffle based warp reduction
 * @param partial_sums  - partial sums within individual threads in the blocks
 */
__inline__ __device__ float4 warp_reduce_sum(float4 partial_sums);


/**
 * CPU implementation of the Center of Mass calculation
 * @param memDesc - Memory descriptor of particle data on CPU side
 */
float4 centerOfMassCPU(MemDesc& memDesc);

/**
 * Check whether the given number is the power of number two or it is not.
 * @param x - Number to check whether it is the power of two
 */
bool ispow2(int x);

/**
 * Wrapper to call kernel for computing the Center Of Mass on GPU.
 * @param particles_gpu     - particles structure
 * @param com               - center of mass structure
 * @param lock              - binary lock
 * @param N                 - number of particles
 * @param reduction_grid    - size of the reduction grid
 * @param red_thr_blc       - number of reduction threads per block
 * @param shm_mem           - required size of the shared memory
 */
void callCenterOfMass(
        t_particles particles_gpu, float4* com, int* lock, const int N,
        const int reduction_grid, const int red_thr_blc, const size_t shm_mem, cudaStream_t stream
);

#endif /* __NBODY_H__ */

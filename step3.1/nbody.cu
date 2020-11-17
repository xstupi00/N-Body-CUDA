/**
 * @File nbody.cu
 *
 * Implementation of the N-Body problem
 *
 * Parallel Computations on GPU (PCG 2020)
 * Assignment no. 1 (cuda)
 * Login: xstupi00
 */

#include <cmath>
#include <cfloat>
#include "nbody.h"

/**
 * CUDA kernel to calculate gravitation and collision velocity and update particles.
 * @param p_in    - input particles
 * @param p_out   - output particles
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_velocity(t_particles p, int N, float dt)
{
    // Declares dynamic allocation of the shared memory
    extern __shared__ float shared_data[];
    int tx = threadIdx.x;   // thread index
    int bdx = blockDim.x;   // block dimensions
    // Computes the global index of thread within the grid
    int i = bdx * blockIdx.x + tx;

    // Checks whether the thread is not outside the particles borders
    // Loads the particle data covered by the individual thread
    float pos_x = (i < N) ? p.pos_x[i] : 0.0f;
    float pos_y = (i < N) ? p.pos_y[i] : 0.0f;
    float pos_z = (i < N) ? p.pos_z[i] : 0.0f;
    float vel_x = (i < N) ? p.vel_x[i] : 0.0f;
    float vel_y = (i < N) ? p.vel_y[i] : 0.0f;
    float vel_z = (i < N) ? p.vel_z[i] : 0.0f;
    float weight_i = (i < N) ? p.weight[i] : 0.0f;

    // Initialises of auxiliary accumulators of velocity
    float tmp_vel_x = 0.0f;
    float tmp_vel_y = 0.0f;
    float tmp_vel_z = 0.0f;

    // Iterates over the number of blocks within the whole grid
    for (int tile = 0; tile < gridDim.x; tile++) {
        // Computes the index to load the shared particle, which will be used by the whole block
        int idx = tile * bdx + tx;
        int tx7 = tx * 7;   // temporary computation

        // Each thread in the block loads one particle from the global memory to shared memory
        // The whole block will have stored the particles data required in the current iteration
        shared_data[tx7] = (idx < N) ? p.pos_x[idx] : 0.0f;
        shared_data[tx7 + 1] = (idx < N) ? p.pos_y[idx] : 0.0f;
        shared_data[tx7 + 2] = (idx < N) ? p.pos_z[idx] : 0.0f;
        shared_data[tx7 + 3] = (idx < N) ? p.vel_x[idx] : 0.0f;
        shared_data[tx7 + 4] = (idx < N) ? p.vel_y[idx] : 0.0f;
        shared_data[tx7 + 5] = (idx < N) ? p.vel_z[idx] : 0.0f;
        shared_data[tx7 + 6] = (idx < N) ? p.weight[idx] : 0.0f;

        // Waits until each thread within block loads the relevant particle to the shared memory
        __syncthreads();

        // Iterates over the particles which were loaded by whole block
        for (int j = 0; j < bdx; j++) {
            int j7 = j * 7; // temporary calculation

            // Computes the distance between the relevant particles
            float r_x = pos_x - shared_data[j7];
            float r_y = pos_y - shared_data[j7 + 1];
            float r_z = pos_z - shared_data[j7 + 2];

            // Loads the weight of the processing particle
            float weight_j = shared_data[j7 + 6];
            // Computes inverse distance between particles and their distances
            float inv_dist = sqrtf(r_x * r_x + r_y * r_y + r_z * r_z);
            // Computes the gravitation velocity (Fg_dt_m2_r)
            float s = weight_j * (-G * dt / (inv_dist * inv_dist * inv_dist + FLT_MIN));

            // The speed that a particle body receives due to the strength of the relevant particle
            tmp_vel_x += (inv_dist > COLLISION_DISTANCE) ? r_x * s : 0.0f;
            tmp_vel_y += (inv_dist > COLLISION_DISTANCE) ? r_y * s : 0.0f;
            tmp_vel_z += (inv_dist > COLLISION_DISTANCE) ? r_z * s : 0.0f;

            // Checks whether the particles are in the sufficient near distance for collision
            if (inv_dist > 0.0f && inv_dist < COLLISION_DISTANCE) {
                // Computes the temporary partial results to eliminate recalculation
                float weight_diff = weight_i - weight_j;
                float weight_sum = weight_i + weight_j;
                float weight_j_x_2 = 2 * weight_j;

                // Computes the collision velocities between the relevant particles and accumulate the results
                tmp_vel_x += inv_dist > 0.0f ?
                             ((weight_diff * vel_x + weight_j_x_2 * shared_data[j7 + 3]) / weight_sum) - vel_x : 0.0f;
                tmp_vel_y += inv_dist > 0.0f ?
                             ((weight_diff * vel_y + weight_j_x_2 * shared_data[j7 + 4]) / weight_sum) - vel_y : 0.0f;
                tmp_vel_z += inv_dist > 0.0f ?
                             ((weight_diff * vel_z + weight_j_x_2 * shared_data[j7 + 5]) / weight_sum) - vel_z : 0.0f;
            }
        }
        // Waits until each thread within the block terminates the calculation in the current iteration
        __syncthreads();
    }

    // Checks whether the thread is not outside the particles borders
    if (i < N) {
        // Updates the velocity of particles with respect to the computed gravitation and collision velocity
        p.vel_x[i] += tmp_vel_x;
        p.vel_y[i] += tmp_vel_y;
        p.vel_z[i] += tmp_vel_z;

        // Updates the positions of particles with respect to the updated velocity
        p.pos_x[i] += p.vel_x[i] * dt;
        p.pos_y[i] += p.vel_y[i] * dt;
        p.pos_z[i] += p.vel_z[i] * dt;
    }

}// end of calculate_velocity
//---------------------------------------------------------------------------------------------------------------------

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
        pos_x[tid] += (p.pos_x[i] * weight_i) + (p.pos_x[i + blockSize] * weight_j);
        pos_y[tid] += (p.pos_y[i] * weight_i) + (p.pos_y[i + blockSize] * weight_j);
        pos_z[tid] += (p.pos_z[i] * weight_i) + (p.pos_z[i + blockSize] * weight_j);
        weights[tid] += weight_i + weight_j;
        // Shift the index for covering the next particle according to the program grid size
        i += gridSize;
    }
    // Synchronizes all threads within the block until they computed the own part of the reduction
    __syncthreads();

    // Iterates over the partial results of performed reduction
    for(unsigned int stride = blockSize / 2; stride > 0; stride >>= 1) {
        // Computes the addition over the relevant particles data according to the requirements of reduction
        pos_x[tid] += (tid < stride) ? pos_x[tid + stride] : 0.0f;
        pos_y[tid] += (tid < stride) ? pos_y[tid + stride] : 0.0f;
        pos_z[tid] += (tid < stride) ? pos_z[tid + stride] : 0.0f;
        weights[tid] += (tid < stride) ? weights[tid + stride] : 0.0f;
        // Synchronizes all threads within the block until they computed the own part of the computation
        __syncthreads();
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
 * @param particles - All particles in the system
 * @param N         - Number of particles
 */
__host__ float4 centerOfMassCPU(MemDesc& memDesc)
{
  float4 com = {0 ,0, 0, 0};

  for(int i = 0; i < memDesc.getDataSize(); i++)
  {
    // Calculate the vector on the line connecting points and most recent position of center-of-mass
    const float dx = memDesc.getPosX(i) - com.x;
    const float dy = memDesc.getPosY(i) - com.y;
    const float dz = memDesc.getPosZ(i) - com.z;

    // Calculate weight ratio only if at least one particle isn't massless
    const float dw = ((memDesc.getWeight(i) + com.w) > 0.0f)
                          ? ( memDesc.getWeight(i) / (memDesc.getWeight(i) + com.w)) : 0.0f;

    // Update position and weight of the center-of-mass according to the weight ration and vector
    com.x += dx * dw;
    com.y += dy * dw;
    com.z += dz * dw;
    com.w += memDesc.getWeight(i);
  }
  return com;
}// enf of centerOfMassCPU
//----------------------------------------------------------------------------------------------------------------------

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
__global__ void calculate_velocity(t_particles p_in, t_particles p_out, int N, float dt)
{
    // Declares dynamic allocation of the shared memory
    extern __shared__ float shared_data[];
    int tx = threadIdx.x;   // thread index
    int tx7 = tx * 7;       // temporary computation
    int bdx = blockDim.x;   // block dimensions
    // Computes the global index of thread within the grid
    int i = bdx * blockIdx.x + tx;

    // Checks whether the thread is not outside the particles borders
    // Loads the particle data covered by the individual thread
    float pos_x = (i < N) ? p_in.pos_x[i] : 0.0f;
    float pos_y = (i < N) ? p_in.pos_y[i] : 0.0f;
    float pos_z = (i < N) ? p_in.pos_z[i] : 0.0f;
    float vel_x = (i < N) ? p_in.vel_x[i] : 0.0f;
    float vel_y = (i < N) ? p_in.vel_y[i] : 0.0f;
    float vel_z = (i < N) ? p_in.vel_z[i] : 0.0f;
    float weight_i = (i < N) ? p_in.weight[i] : 0.0f;

    // Initialises of auxiliary accumulators of velocity
    float tmp_vel_x = 0.0f;
    float tmp_vel_y = 0.0f;
    float tmp_vel_z = 0.0f;

    // Iterates over the number of blocks within the whole grid
    for (int tile = 0; tile < gridDim.x; tile++) {
        // Computes the index to load the shared particle, which will be used by the whole block
        int idx = tile * bdx + tx;

        // Each thread in the block loads one particle from the global memory to shared memory
        // The whole block will have stored the particles data required in the current iteration
        shared_data[tx7] = (idx < N) ? p_in.pos_x[idx] : 0.0f;
        shared_data[tx7 + 1] = (idx < N) ? p_in.pos_y[idx] : 0.0f;
        shared_data[tx7 + 2] = (idx < N) ? p_in.pos_z[idx] : 0.0f;
        shared_data[tx7 + 3] = (idx < N) ? p_in.vel_x[idx] : 0.0f;
        shared_data[tx7 + 4] = (idx < N) ? p_in.vel_y[idx] : 0.0f;
        shared_data[tx7 + 5] = (idx < N) ? p_in.vel_z[idx] : 0.0f;
        shared_data[tx7 + 6] = (idx < N) ? p_in.weight[idx] : 0.0f;

        // Waits until each thread within block loads the relevant particle to the shared memory
        __syncthreads();

        // Iterates over the particles which were loaded by whole block
        for (int j = 0; j < bdx; j++) {
            int j7 = j * 7; // temporary calculation

            // Loads the weight of the processing particle
            float weight_j = shared_data[j7 + 6];
            // Instruction Level Parallelism
            float s = -G * dt * weight_j;

            // Computes the distance between the relevant particles
            float r_x = pos_x - shared_data[j7];
            float r_y = pos_y - shared_data[j7 + 1];
            float r_z = pos_z - shared_data[j7 + 2];

            // Computes inverse distance between particles and their distances
            float inv_dist = sqrtf(r_x * r_x + r_y * r_y + r_z * r_z);
            // Computes the gravitation velocity (Fg_dt_m2_r)
            s /= (inv_dist * inv_dist * inv_dist + FLT_MIN);

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
        p_out.vel_x[i] = vel_x + tmp_vel_x;
        p_out.vel_y[i] = vel_y + tmp_vel_y;
        p_out.vel_z[i] = vel_z + tmp_vel_z;

        // Updates the positions of particles with respect to the updated velocity
        p_out.pos_x[i] = pos_x + p_out.vel_x[i] * dt;
        p_out.pos_y[i] = pos_y + p_out.vel_y[i] * dt;
        p_out.pos_z[i] = pos_z + p_out.vel_z[i] * dt;
    }

}// end of calculate_velocity
//---------------------------------------------------------------------------------------------------------------------


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


/**
 * CUDA kernel function to perform shuffle based warp reduction
 * @param partial_sums  - partial sums within individual threads in the blocks
 */
__inline__ __device__
float4 warp_reduce_sum(float4 partial_sums) {
    // Compute the reduction tree by using the shuffle down
    // All threads will be shifting values even though they are not needed in the reduction
    for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2) { // stride => 16, 8, 4, 2, 1
        partial_sums.x += __shfl_down_sync(0xffffffff, partial_sums.x, stride);
        partial_sums.y += __shfl_down_sync(0xffffffff, partial_sums.y, stride);
        partial_sums.z += __shfl_down_sync(0xffffffff, partial_sums.z, stride);
        partial_sums.w += __shfl_down_sync(0xffffffff, partial_sums.w, stride);
    }
    // After executing the three reduction thread 0 has the total reduced value in its relevant partial sums
    return partial_sums;
}// end of warp_reduce_sum
//---------------------------------------------------------------------------------------------------------------------


/**
 * CUDA kernel function to perform reduction across the entire block
 * @param partial_sums  - partial sums within individual threads in the blocks
 */
__inline__ __device__
float4 block_reduce_sum(float4 partial_sums)
{
    // Declares dynamic allocation of the shared memory
    extern __shared__ float shared_data[];
    // Computes the number of partial sums calculated by individual warps within one block
    int partial_sum_count = blockDim.x / WARP_SIZE;
    // Obtains the pointers to the relevant parts of the shared memory for individual items of particles data
    float* pos_x = shared_data;
    float* pos_y = &shared_data[partial_sum_count];
    float* pos_z = &shared_data[partial_sum_count * 2];
    float* weights = &shared_data[partial_sum_count * 3];

    // Compute the thread's index within its warp - from 0 to 31
    int lane = threadIdx.x % WARP_SIZE;
    // Compute the warp's index within whole block
    int wid = threadIdx.x / WARP_SIZE;

    // Each warp performs partial reduction
    partial_sums = warp_reduce_sum(partial_sums);

    // Write reduce value by each warp to shared memory
    if (lane == 0) {    // 0.thread in the each warp
        pos_x[wid] = partial_sums.x;    // position x
        pos_y[wid] = partial_sums.y;    // position y
        pos_z[wid] = partial_sums.z;    // position z
        weights[wid] = partial_sums.w;  // weight
    }
    // Synchronizes all threads within the block until they computed the own part of the computation
    __syncthreads();

    // Read from shared memory only if that warp existed
    partial_sums.x = (threadIdx.x < blockDim.x / WARP_SIZE) ? pos_x[lane] : 0.0f;
    partial_sums.y = (threadIdx.x < blockDim.x / WARP_SIZE) ? pos_y[lane] : 0.0f;
    partial_sums.z = (threadIdx.x < blockDim.x / WARP_SIZE) ? pos_z[lane] : 0.0f;
    partial_sums.w = (threadIdx.x < blockDim.x / WARP_SIZE) ? weights[lane] : 0.0f;

    // Final reduce within the first warp - warp with index 0 in the each block
    if (wid == 0) {
        partial_sums = warp_reduce_sum(partial_sums);
    }

    // Return partial sum computed within the whole block
    return partial_sums;
}// end of block_reduce_sum
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
    // Initializations the sums within the registers for each thread
    // x (pos_x), y (pos_y), z (pos_z), w (weights)
    float4 sums = {0.0f, 0.0f, 0.0f, 0.0f};

    // Reduce multiple particles per thread
    // Each thread starts at the own thread index and shifts about the overall number of threads in the grid
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N; idx += blockDim.x * gridDim.x) {
        // Loads the weights of particle to reduce the duplicate readings from the memory
        float weight_i = p.weight[idx];
        // Computes the product of relevant position and weight and subsequent addition to the thread registers sums
        sums.x += p.pos_x[idx] * weight_i;  // position x
        sums.y += p.pos_y[idx] * weight_i;  // position y
        sums.z += p.pos_z[idx] * weight_i;  // position z
        sums.w += p.weight[idx];            // weight
    }

    // Perform the reduction across the entire block
    sums = block_reduce_sum(sums);

    // Thread with index 0 within the block writes the result of the reduction to the GPU memory after the computation
    if (threadIdx.x == 0) {
        // Ensures the mutual exclusion for reducing the result to the global memory between the different blocks
        while (0 != atomicCAS(lock, 0, 1)) {}
        // Writes the individual results of the reduction to the final results into global memory GPU
        *comX += sums.x;
        *comY += sums.y;
        *comZ += sums.z;
        *comW += sums.w;
        // Returns the lock to write to the shared global memory GPU
        atomicExch(lock, 0);
    }

}// end of centerOfMass
//----------------------------------------------------------------------------------------------------------------------

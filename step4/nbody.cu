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
 * Check whether the given numner is the power of number two or it is not.
 * @param x - Number to check whether it is the power of two
 */
bool ispow2(int x) { return !((~(~0U>>1)|x)&x -1) ; }

/**
 * Ensures the invocation of kernel to compute COM on the GPU according to required templates.
 * @param p         - particles
 * @param com       - pointer to Center of Mass structure
 * @param lock      - binary lock
 * @param N         - number of particles
 * @param red_grid  - size of the reduction grid
 * @param red_thr   - number of reduction threads per block
 * @param shm_mem   - required size of the shared memory
 * @param m_stream  - CUDA stream in which will be executed the compute kernel
 */
void compute_gpu_center_of_mass(
        t_particles p, float4* com, int* lock, const int N, const int red_grid,
        const int red_thr, size_t shm_mem, cudaStream_t* m_stream)
{
    if (ispow2(N)) { // Check whether the number of particles is power of two
        switch (red_thr) { // Switch between the all possible numbers of reduction threads
            case 1024:
                centerOfMass<1024, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 512:
                centerOfMass<512, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 256:
                centerOfMass<256, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 128:
                centerOfMass<128, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 64:
                centerOfMass<64, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 32:
                centerOfMass<32, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 16:
                centerOfMass<16, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 8:
                centerOfMass<8, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 4:
                centerOfMass<4, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 2:
                centerOfMass<2, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 1:
                centerOfMass<1, true> << < red_grid, red_thr, shm_mem, *m_stream >> >
                      (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;
        }
    } else { // The number of particles is NOT power of two!
        switch (red_thr) {
            case 1024:
                centerOfMass<1024, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);

                break;

            case 512:
                centerOfMass<512, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 256:
                centerOfMass<256, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);

                break;

            case 128:
                centerOfMass<128, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 64:
                centerOfMass<64, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 32:
                centerOfMass<32, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 16:
                centerOfMass<16, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 8:
                centerOfMass<8, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 4:
                centerOfMass<4, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 2:
                centerOfMass<2, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;

            case 1:
                centerOfMass<1, false> <<< red_grid, red_thr, shm_mem, *m_stream >>>
                        (p, &(*com).x, &(*com).y, &(*com).z, &(*com).w, &(*lock), N);
                break;
        } // end switch
    }  // end if-else
}

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

__global__ void calculate_velocity(t_particles p_in, t_particles p_out, int N, float dt)
{
    extern __shared__ float shared_data[];
    int tx = threadIdx.x;
    int bdx = blockDim.x;
    int i = bdx * blockIdx.x + tx;

    float pos_x = (i < N) ? p_in.pos_x[i] : 0.0f;
    float pos_y = (i < N) ? p_in.pos_y[i] : 0.0f;
    float pos_z = (i < N) ? p_in.pos_z[i] : 0.0f;
    float vel_x = (i < N) ? p_in.vel_x[i] : 0.0f;
    float vel_y = (i < N) ? p_in.vel_y[i] : 0.0f;
    float vel_z = (i < N) ? p_in.vel_z[i] : 0.0f;
    float weight_i = (i < N) ? p_in.weight[i] : 0.0f;

    float tmp_vel_x = 0.0f;
    float tmp_vel_y = 0.0f;
    float tmp_vel_z = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
        int idx = tile * bdx + tx;
        int tx7 = tx * 7;

        shared_data[tx7] = (idx < N) ? p_in.pos_x[idx] : 0.0f;
        shared_data[tx7 + 1] = (idx < N) ? p_in.pos_y[idx] : 0.0f;
        shared_data[tx7 + 2] = (idx < N) ? p_in.pos_z[idx] : 0.0f;
        shared_data[tx7 + 3] = (idx < N) ? p_in.vel_x[idx] : 0.0f;
        shared_data[tx7 + 4] = (idx < N) ? p_in.vel_y[idx] : 0.0f;
        shared_data[tx7 + 5] = (idx < N) ? p_in.vel_z[idx] : 0.0f;
        shared_data[tx7 + 6] = (idx < N) ? p_in.weight[idx] : 0.0f;

        __syncthreads();

        for (int j = 0; j < bdx; j++) {
            int j7 = j * 7;

            float r_x = pos_x - shared_data[j7];
            float r_y = pos_y - shared_data[j7 + 1];
            float r_z = pos_z - shared_data[j7 + 2];

            float weight_j = shared_data[j7 + 6];
            float inv_dist = sqrtf(r_x * r_x + r_y * r_y + r_z * r_z);
            float s = weight_j * (-G * dt / (inv_dist * inv_dist * inv_dist + FLT_MIN));

            tmp_vel_x += (inv_dist > COLLISION_DISTANCE) ? r_x * s : 0.0f;
            tmp_vel_y += (inv_dist > COLLISION_DISTANCE) ? r_y * s : 0.0f;
            tmp_vel_z += (inv_dist > COLLISION_DISTANCE) ? r_z * s : 0.0f;

            if (inv_dist > 0.0f && inv_dist < COLLISION_DISTANCE) {

                float weight_diff = weight_i - weight_j;
                float weight_sum = weight_i + weight_j;
                float weight_j_x_2 = 2 * weight_j;

                tmp_vel_x += inv_dist > 0.0f ?
                             ((weight_diff * vel_x + weight_j_x_2 * shared_data[j7 + 3]) / weight_sum) - vel_x : 0.0f;
                tmp_vel_y += inv_dist > 0.0f ?
                             ((weight_diff * vel_y + weight_j_x_2 * shared_data[j7 + 4]) / weight_sum) - vel_y : 0.0f;
                tmp_vel_z += inv_dist > 0.0f ?
                             ((weight_diff * vel_z + weight_j_x_2 * shared_data[j7 + 5]) / weight_sum) - vel_z : 0.0f;
            }
        }
        __syncthreads();
    }

    if (i < N) {
        p_out.vel_x[i] = vel_x + tmp_vel_x;
        p_out.vel_y[i] = vel_y + tmp_vel_y;
        p_out.vel_z[i] = vel_z + tmp_vel_z;

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

bool ispow2(int x) { return !((~(~0U>>1)|x)&x -1) ; }

void compute_gpu_center_of_mass(
        t_particles p, float4* com, int* lock, const int N, const int red_grid,
        const int red_thr, size_t shm_mem, cudaStream_t* m_stream)
{
    if (ispow2(N)) {
        switch (red_thr) {
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
    } else {
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
        }
    }
}

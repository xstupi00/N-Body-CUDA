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

__global__ void calculate_velocity(t_particles p, int N, float dt)
{
    extern __shared__ float shared_data[];
    int tx = threadIdx.x;
    int bdx = blockDim.x;
    int i = bdx * blockIdx.x + tx;

    float pos_x = (i < N) ? p.pos_x[i] : 0.0f;
    float pos_y = (i < N) ? p.pos_y[i] : 0.0f;
    float pos_z = (i < N) ? p.pos_z[i] : 0.0f;
    float vel_x = (i < N) ? p.vel_x[i] : 0.0f;
    float vel_y = (i < N) ? p.vel_y[i] : 0.0f;
    float vel_z = (i < N) ? p.vel_z[i] : 0.0f;
    float weight_i = (i < N) ? p.weight[i] : 0.0f;

    float tmp_vel_x = 0.0f;
    float tmp_vel_y = 0.0f;
    float tmp_vel_z = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
        int idx = tile * bdx + tx;
        int tx7 = tx * 7;

        shared_data[tx7] = (idx < N) ? p.pos_x[idx] : 0.0f;
        shared_data[tx7 + 1] = (idx < N) ? p.pos_y[idx] : 0.0f;
        shared_data[tx7 + 2] = (idx < N) ? p.pos_z[idx] : 0.0f;
        shared_data[tx7 + 3] = (idx < N) ? p.vel_x[idx] : 0.0f;
        shared_data[tx7 + 4] = (idx < N) ? p.vel_y[idx] : 0.0f;
        shared_data[tx7 + 5] = (idx < N) ? p.vel_z[idx] : 0.0f;
        shared_data[tx7 + 6] = (idx < N) ? p.weight[idx] : 0.0f;

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
        p.vel_x[i] += tmp_vel_x;
        p.vel_y[i] += tmp_vel_y;
        p.vel_z[i] += tmp_vel_z;

        p.pos_x[i] += p.vel_x[i] * dt;
        p.pos_y[i] += p.vel_y[i] * dt;
        p.pos_z[i] += p.vel_z[i] * dt;
    }

}// end of calculate_velocity
//---------------------------------------------------------------------------------------------------------------------

__device__ void warpReduce(
        volatile float* pos_x, volatile float* pos_y, volatile float* pos_z,
        volatile float* weights, unsigned int tid, unsigned int blockSize
) {
    if (blockSize >= 64) {
        pos_x[tid] += pos_x[tid + 32];
        pos_y[tid] += pos_y[tid + 32];
        pos_z[tid] += pos_z[tid + 32];
        weights[tid] += weights[tid + 32];
    }
    __syncwarp();

    if (blockSize >= 32) {
        pos_x[tid] += pos_x[tid + 16];
        pos_y[tid] += pos_y[tid + 16];
        pos_z[tid] += pos_z[tid + 16];
        weights[tid] += weights[tid + 16];
    }
    __syncwarp();

    if (blockSize >= 16) {
        pos_x[tid] += pos_x[tid + 8];
        pos_y[tid] += pos_y[tid + 8];
        pos_z[tid] += pos_z[tid + 8];
        weights[tid] += weights[tid + 8];
    }
    __syncwarp();

    if (blockSize >= 8) {
        pos_x[tid] += pos_x[tid + 4];
        pos_y[tid] += pos_y[tid + 4];
        pos_z[tid] += pos_z[tid + 4];
        weights[tid] += weights[tid + 4];
    }
    __syncwarp();

    if (blockSize >= 4) {
        pos_x[tid] += pos_x[tid + 2];
        pos_y[tid] += pos_y[tid + 2];
        pos_z[tid] += pos_z[tid + 2];
        weights[tid] += weights[tid + 2];
    }
    __syncwarp();

    if (blockSize >= 2) {
        pos_x[tid] += pos_x[tid + 1];
        pos_y[tid] += pos_y[tid + 1];
        pos_z[tid] += pos_z[tid + 1];
        weights[tid] += weights[tid + 1];
    }
    __syncwarp();

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
__global__ void centerOfMass(t_particles p, float* comX, float* comY, float* comZ, float* comW, int* lock, const int N)
{
    extern __shared__ float shared_data[];
    unsigned int blockSize = blockDim.x;
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
        pos_x[tid] += (p.pos_x[i] * weight_i) + (p.pos_x[i + blockSize] * weight_j);
        pos_y[tid] += (p.pos_y[i] * weight_i) + (p.pos_y[i + blockSize] * weight_j);
        pos_z[tid] += (p.pos_z[i] * weight_i) + (p.pos_z[i + blockSize] * weight_j);
        weights[tid] += weight_i + weight_j;
        i += gridSize;
    }
    __syncthreads();

    for(unsigned int stride = blockSize / 2; stride > 32; stride >>= 1) {
        pos_x[tid] += (tid < stride) ? pos_x[tid + stride] : 0.0f;
        pos_y[tid] += (tid < stride) ? pos_y[tid + stride] : 0.0f;
        pos_z[tid] += (tid < stride) ? pos_z[tid + stride] : 0.0f;
        weights[tid] += (tid < stride) ? weights[tid + stride] : 0.0f;
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(pos_x, pos_y, pos_z, weights, tid, blockSize);
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

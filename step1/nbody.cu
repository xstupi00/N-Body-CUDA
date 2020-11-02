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
__global__ void calculate_velocity(
        const t_particles p_in, t_particles p_out, int N, float dt
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {

        float pos_x = p_in.pos_x[i];
        float pos_y = p_in.pos_y[i];
        float pos_z = p_in.pos_z[i];
        float vel_x = p_in.vel_x[i];
        float vel_y = p_in.vel_y[i];
        float vel_z = p_in.vel_z[i];
        float weight_i = p_in.weight[i];

        float tmp_vel_x = 0.0f;
        float tmp_vel_y = 0.0f;
        float tmp_vel_z = 0.0f;

        for (int j = 0; j < N; j++) {

            float r_x = pos_x - p_in.pos_x[j];
            float r_y = pos_y - p_in.pos_y[j];
            float r_z = pos_z - p_in.pos_z[j];

            float weight_j = p_in.weight[j];
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
                             ((weight_diff * vel_x + weight_j_x_2 * p_in.vel_x[j]) / weight_sum) - vel_x : 0.0f;
                tmp_vel_y += inv_dist > 0.0f ?
                             ((weight_diff * vel_y + weight_j_x_2 * p_in.vel_y[j]) / weight_sum) - vel_y : 0.0f;
                tmp_vel_z += inv_dist > 0.0f ?
                             ((weight_diff * vel_z + weight_j_x_2 * p_in.vel_z[j]) / weight_sum) - vel_z : 0.0f;
            }
        }

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

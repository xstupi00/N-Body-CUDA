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
 * CUDA kernel to calculate gravitation velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_gravitation_velocity(t_particles p, t_velocities tmp_vel, int N, float dt)
{
    // Computes the global index of thread within the grid
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Checks whether the thread is not outside the particles borders
    if (i < N) {

        // Loads the particle data covered by the individual thread
        float pos_x = p.pos_x[i];
        float pos_y = p.pos_y[i];
        float pos_z = p.pos_z[i];

        // Initialise of auxiliary accumulators of velocity
        float tmp_vel_x = 0.0f;
        float tmp_vel_y = 0.0f;
        float tmp_vel_z = 0.0f;

        // The iterations over all particles to compute the gravitation velocity to them
        for (int j = 0; j < N; j++) {

            // Instruction Level Parallelism
            float s = -G * dt * p.weight[j];
            // Computes the distance between the relevant particles
            float r_x = pos_x - p.pos_x[j];
            float r_y = pos_y - p.pos_y[j];
            float r_z = pos_z - p.pos_z[j];

            // Computes inverse distance between particles and their distances
            float inv_dist = sqrtf(r_x * r_x + r_y * r_y + r_z * r_z);
            // Computes the gravitation velocity (Fg_dt_m2_r)
            s /= (inv_dist * inv_dist * inv_dist + FLT_MIN);

            // The speed that a particle body receives due to the strength of the relevant particle
            tmp_vel_x += (inv_dist > COLLISION_DISTANCE) ? r_x * s : 0.0f;
            tmp_vel_y += (inv_dist > COLLISION_DISTANCE) ? r_y * s : 0.0f;
            tmp_vel_z += (inv_dist > COLLISION_DISTANCE) ? r_z * s : 0.0f;
        }

        // Stores the final computed velocities of each particle to the auxiliary velocity vector
        tmp_vel.x[i] = tmp_vel_x;
        tmp_vel.y[i] = tmp_vel_y;
        tmp_vel.z[i] = tmp_vel_z;
    }

}// end of calculate_gravitation_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to calculate collision velocity
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void calculate_collision_velocity(t_particles p, t_velocities tmp_vel, int N, float dt)
{
    // Computes the global index of thread within the grid
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Checks whether the thread is not outside the particles borders
    if (i < N) {
        // Loads the particle data covered by the individual thread
        float pos_x = p.pos_x[i];
        float pos_y = p.pos_y[i];
        float pos_z = p.pos_z[i];
        float vel_x = p.vel_x[i];
        float vel_y = p.vel_y[i];
        float vel_z = p.vel_z[i];
        float weight_i = p.weight[i];

        // Initialise of auxiliary accumulators of velocity
        float tmp_vel_x = 0.0f;
        float tmp_vel_y = 0.0f;
        float tmp_vel_z = 0.0f;

        // The iterations over all particles to compute the collision velocity to them
        for (int j = 0; j < N; j++) {

            // Computes the distance between the relevant particles
            float r_x = pos_x - p.pos_x[j];
            float r_y = pos_y - p.pos_y[j];
            float r_z = pos_z - p.pos_z[j];

            // Loads the weight of the processing particle
            float weight_j = p.weight[j];
            // Computes inverse distance between particles and their distances
            float inv_dist = sqrtf(r_x * r_x + r_y * r_y + r_z * r_z);

            // Checks whether the particles are in the sufficient near distance for collision
            if (inv_dist > 0.0f && inv_dist < COLLISION_DISTANCE) {
                // Computes the temporary partial results to eliminate recalculation
                float weight_diff = weight_i - weight_j;
                float weight_sum = weight_i + weight_j;
                float weight_j_x_2 = 2 * weight_j;

                // Computes the collision velocities between the relevant particles and accumulate the results
                tmp_vel_x += inv_dist > 0.0f ?
                             ((weight_diff * vel_x + weight_j_x_2 * p.vel_x[j]) / weight_sum) - vel_x : 0.0f;
                tmp_vel_y += inv_dist > 0.0f ?
                             ((weight_diff * vel_y + weight_j_x_2 * p.vel_y[j]) / weight_sum) - vel_y : 0.0f;
                tmp_vel_z += inv_dist > 0.0f ?
                             ((weight_diff * vel_z + weight_j_x_2 * p.vel_z[j]) / weight_sum) - vel_z : 0.0f;
            }
        }

        // Stores the final computed velocities of each particle to the auxiliary velocity vector
        tmp_vel.x[i] += tmp_vel_x;
        tmp_vel.y[i] += tmp_vel_y;
        tmp_vel.z[i] += tmp_vel_z;
    }

}// end of calculate_collision_velocity
//----------------------------------------------------------------------------------------------------------------------

/**
 * CUDA kernel to update particles
 * @param p       - particles
 * @param tmp_vel - temp array for velocities
 * @param N       - Number of particles
 * @param dt      - Size of the time step
 */
__global__ void update_particle(t_particles p, t_velocities tmp_vel, int N, float dt)
{
    // Computes the global index of thread within the grid
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // Checks whether the thread is not outside the particles borders
    if (i < N) {
        // Updates the velocity of particles with respect to the computed gravitation and collision velocity
        p.vel_x[i] += tmp_vel.x[i];
        p.vel_y[i] += tmp_vel.y[i];
        p.vel_z[i] += tmp_vel.z[i];

        // Updates the positions of particles with respect to the updated velocity
        p.pos_x[i] += p.vel_x[i] * dt;
        p.pos_y[i] += p.vel_y[i] * dt;
        p.pos_z[i] += p.vel_z[i] * dt;
    }

}// end of update_particle
//----------------------------------------------------------------------------------------------------------------------

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

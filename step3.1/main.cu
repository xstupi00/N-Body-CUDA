/**
 * @File main.cu
 *
 * The main file of the project
 *
 * Parallel Computations on GPU (PCG 2020)
 * Assignment no. 1 (cuda)
 * Login: xstupi00
 */

#include <sys/time.h>
#include <cstdio>
#include <cmath>
#include <vector>

#include <helper_cuda.h>

#include "nbody.h"
#include "h5Helper.h"

/**
 * Main rotine
 * @param argc
 * @param argv
 * @return
 */
int main(int argc, char **argv) {
    // Time measurement
    struct timeval t1, t2;

    if (argc != 10) {
        printf("Usage: "
               "nbody <N> <dt> <steps> <threads/block> <write intesity> "
               "<reduction threads> <reduction threads/block> <input> <output>\n"
        );
        exit(1);
    }

    // Number of particles
    const int N = std::stoi(argv[1]);
    // Length of time step
    const float dt = std::stof(argv[2]);
    // Number of steps
    const int steps = std::stoi(argv[3]);
    // Number of thread blocks
    const int thr_blc = std::stoi(argv[4]);
    // Write frequency
    int writeFreq = std::stoi(argv[5]);
    // number of reduction threads
    const int red_thr = std::stoi(argv[6]);
    // Number of reduction threads/blocks
    const int red_thr_blc = std::stoi(argv[7]);

    // Size of the simulation CUDA gird - number of blocks
    const size_t simulationGrid = (N + thr_blc - 1) / thr_blc;
    // Size of the reduction CUDA grid - number of blocks
    const size_t reductionGrid = (red_thr + red_thr_blc - 1) / red_thr_blc;

    // Log benchmark setup
    printf("N: %d\n", N);
    printf("dt: %f\n", dt);
    printf("steps: %d\n", steps);
    printf("threads/block: %d\n", thr_blc);
    printf("blocks/grid: %lu\n", simulationGrid);
    printf("reduction threads/block: %d\n", red_thr_blc);
    printf("reduction blocks/grid: %lu\n", reductionGrid);

    // Number of records to continuous writing of partial results
    const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
    writeFreq = (writeFreq > 0) ? writeFreq : 0;

    // CPU particles structures
    t_particles particles_cpu;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                            FILL IN: CPU side memory allocation (step 0)                                        //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // The overall memory size of input particles
    size_t size = N * sizeof(float);
    // Allocates page-locked memory on the host. Maps the allocation into the CUDA address space
    checkCudaErrors(cudaHostAlloc(&particles_cpu.pos_x, size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc(&particles_cpu.pos_y, size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc(&particles_cpu.pos_z, size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc(&particles_cpu.vel_x, size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc(&particles_cpu.vel_y, size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc(&particles_cpu.vel_z, size, cudaHostAllocMapped));
    checkCudaErrors(cudaHostAlloc(&particles_cpu.weight, size, cudaHostAllocMapped));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                              FILL IN: memory layout descriptor (step 0)                                        //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    /*
     * Caution! Create only after CPU side allocation
     * parameters:
     *                      Stride of two               Offset of the first
     *  Data pointer        consecutive elements        element in floats,
     *                      in floats, not bytes        not bytes
    */
    MemDesc md(
            particles_cpu.pos_x, 1, 0,              // Position in X
            particles_cpu.pos_y, 1, 0,              // Position in Y
            particles_cpu.pos_z, 1, 0,              // Position in Z
            particles_cpu.vel_x, 1, 0,              // Velocity in X
            particles_cpu.vel_y, 1, 0,              // Velocity in Y
            particles_cpu.vel_z, 1, 0,              // Velocity in Z
            particles_cpu.weight, 1, 0,             // Weight
            N,                                      // Number of particles
            recordsNum                              // Number of records in output file
    );

    // Initialisation of helper class and loading of input data
    H5Helper h5Helper(argv[8], argv[9], md);

    try {
        h5Helper.init();
        h5Helper.readParticleData();
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                  FILL IN: GPU side memory allocation (step 0)                                  //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // GPU particles structure
    t_particles particles_gpu;

    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&particles_gpu.pos_x, size));
    checkCudaErrors(cudaMalloc(&particles_gpu.pos_y, size));
    checkCudaErrors(cudaMalloc(&particles_gpu.pos_z, size));
    checkCudaErrors(cudaMalloc(&particles_gpu.vel_x, size));
    checkCudaErrors(cudaMalloc(&particles_gpu.vel_y, size));
    checkCudaErrors(cudaMalloc(&particles_gpu.vel_z, size));
    checkCudaErrors(cudaMalloc(&particles_gpu.weight, size));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       FILL IN: memory transfers (step 0)                                       //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Copies particles data from host to device.
    checkCudaErrors(cudaMemcpy(particles_gpu.pos_x, particles_cpu.pos_x, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu.pos_y, particles_cpu.pos_y, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu.pos_z, particles_cpu.pos_z, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu.vel_x, particles_cpu.vel_x, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu.vel_y, particles_cpu.vel_y, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu.vel_z, particles_cpu.vel_z, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu.weight, particles_cpu.weight, size, cudaMemcpyHostToDevice));

    gettimeofday(&t1, 0);

    for (int s = 0; s < steps; s++) {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                       FILL IN: kernels invocation (step 0)                                 //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        // Compute the size of the shared memory (for one grid block)
        size_t shm_mem = thr_blc * sizeof(float) * 7;
        calculate_velocity<<<simulationGrid, thr_blc, shm_mem>>>(particles_gpu, N, dt);

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                          FILL IN: synchronization  (step 4)                                    //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        if (writeFreq > 0 && (s % writeFreq == 0)) {
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                          FILL IN: synchronization and file access logic (step 4)                             //
            //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              FILL IN: invocation of center-of-mass kernel (step 3.1, step 3.2, step 4)                         //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // CPU Center of Mass structure - x, y, z, w
    float4* comCPU;
    // GPU Center of Mass structure - x, y, z, w
    float4* comGPU;
    // Lock to ensure the mutual exclusion for reducing the result to the global memory
    int* lock;

    // Allocates page-locked memory on the host
    checkCudaErrors(cudaHostAlloc(&comCPU, sizeof(float4), cudaHostAllocMapped));
    // Allocate memory on the device
    checkCudaErrors(cudaMalloc(&comGPU, sizeof(float4)));
    checkCudaErrors(cudaMalloc(&lock, sizeof(int)));

    // Initializes or sets device memory to a zero value
    checkCudaErrors(cudaMemset(comCPU, 0, sizeof(float4)));
    checkCudaErrors(cudaMemset(comGPU, 0, sizeof(float4)));
    checkCudaErrors(cudaMemset(lock, 0, sizeof(int)));

    // Computes the size of the shared memory for reduction block
    size_t shm_mem = red_thr_blc * sizeof(float) * 4;
    // Calls reduction kernel to compute the Center of Mass
    if (ispow2(N)) {
        centerOfMass <true> <<<reductionGrid, red_thr_blc, shm_mem>>> (
            particles_gpu, &comGPU[0].x, &comGPU[0].y, &comGPU[0].z, &comGPU[0].w, &lock[0], N
        );
    } else {
        centerOfMass <false> <<<reductionGrid, red_thr_blc, shm_mem>>> (
                particles_gpu, &comGPU[0].x, &comGPU[0].y, &comGPU[0].z, &comGPU[0].w, &lock[0], N
        );
    }

    cudaDeviceSynchronize();

    gettimeofday(&t2, 0);

    // Approximate simulation wall time
    double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("Time: %f s\n", t);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                             FILL IN: memory transfers for particle data (step 0)                                 //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Copies particles data from device to host
    checkCudaErrors(cudaMemcpy(particles_cpu.pos_x, particles_gpu.pos_x, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.pos_y, particles_gpu.pos_y, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.pos_z, particles_gpu.pos_z, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.vel_x, particles_gpu.vel_x, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.vel_y, particles_gpu.vel_y, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.vel_z, particles_gpu.vel_z, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.weight, particles_gpu.weight, size, cudaMemcpyDeviceToHost));
    // Copies Center of Mass data from device to host
    checkCudaErrors(cudaMemcpy(comCPU, comGPU, sizeof(float4), cudaMemcpyDeviceToHost));

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                        FILL IN: memory transfers for center-of-mass (step 3.1, step 3.2)                         //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    float4 comOnCPU = centerOfMassCPU(md);

    std::cout << "Center of mass on CPU:" << std::endl
              << comOnCPU.x << ", "
              << comOnCPU.y << ", "
              << comOnCPU.z << ", "
              << comOnCPU.w
              << std::endl;

    std::cout << "Center of mass on GPU:" << std::endl
              << comCPU[0].x / comCPU[0].w  << ", "
              << comCPU[0].y / comCPU[0].w << ", "
              << comCPU[0].z / comCPU[0].w << ", "
              << comCPU[0].w
              << std::endl;

    // Writing final values to the file
    h5Helper.writeComFinal(comCPU[0].x, comCPU[0].y, comCPU[0].z, comCPU[0].w);
    h5Helper.writeParticleDataFinal();

    // Free page-locked memory.
    checkCudaErrors(cudaFreeHost(particles_cpu.pos_x));
    checkCudaErrors(cudaFreeHost(particles_cpu.pos_y));
    checkCudaErrors(cudaFreeHost(particles_cpu.pos_z));
    checkCudaErrors(cudaFreeHost(particles_cpu.vel_x));
    checkCudaErrors(cudaFreeHost(particles_cpu.vel_y));
    checkCudaErrors(cudaFreeHost(particles_cpu.vel_z));
    checkCudaErrors(cudaFreeHost(particles_cpu.weight));
    checkCudaErrors(cudaFreeHost(comCPU));

    // Free memory on the device.
    checkCudaErrors(cudaFree(particles_gpu.pos_x));
    checkCudaErrors(cudaFree(particles_gpu.pos_y));
    checkCudaErrors(cudaFree(particles_gpu.pos_z));
    checkCudaErrors(cudaFree(particles_gpu.vel_x));
    checkCudaErrors(cudaFree(particles_gpu.vel_y));
    checkCudaErrors(cudaFree(particles_gpu.vel_z));
    checkCudaErrors(cudaFree(particles_gpu.weight));
    checkCudaErrors(cudaFree(comGPU));
    checkCudaErrors(cudaFree(lock));

    return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

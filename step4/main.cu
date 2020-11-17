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
        printf("Usage: nbody <N> <dt> <steps> <threads/block> <write intesity> <reduction threads> <reduction threads/block> <input> <output>\n");
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

    const size_t recordsNum = (writeFreq > 0) ? (steps + writeFreq - 1) / writeFreq : 0;
    writeFreq = (writeFreq > 0) ? writeFreq : 0;


    t_particles particles_cpu;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                            FILL IN: CPU side memory allocation (step 0)                                        //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    size_t size = N * sizeof(float);
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
    std::vector<t_particles> particles_gpu(2);

    for (auto &p_gpu : particles_gpu) {
        checkCudaErrors(cudaMalloc(&p_gpu.pos_x, size));
        checkCudaErrors(cudaMalloc(&p_gpu.pos_y, size));
        checkCudaErrors(cudaMalloc(&p_gpu.pos_z, size));
        checkCudaErrors(cudaMalloc(&p_gpu.vel_x, size));
        checkCudaErrors(cudaMalloc(&p_gpu.vel_y, size));
        checkCudaErrors(cudaMalloc(&p_gpu.vel_z, size));
        checkCudaErrors(cudaMalloc(&p_gpu.weight, size));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                       FILL IN: memory transfers (step 0)                                       //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    checkCudaErrors(cudaMemcpy(particles_gpu[0].pos_x, particles_cpu.pos_x, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu[0].pos_y, particles_cpu.pos_y, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu[0].pos_z, particles_cpu.pos_z, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu[0].vel_x, particles_cpu.vel_x, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu[0].vel_y, particles_cpu.vel_y, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu[0].vel_z, particles_cpu.vel_z, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu[0].weight, particles_cpu.weight, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(particles_gpu[1].weight, particles_gpu[0].weight, size, cudaMemcpyDeviceToDevice));

    float4* comCPU;
    float4* comGPU;
    int* lock;

    checkCudaErrors(cudaHostAlloc(&comCPU, sizeof(float4), cudaHostAllocMapped));
    checkCudaErrors(cudaMalloc(&comGPU, sizeof(float4)));
    checkCudaErrors(cudaMalloc(&lock, sizeof(int)));

    checkCudaErrors(cudaMemset(comCPU, 0, sizeof(float4)));
    checkCudaErrors(cudaMemset(lock, 0, sizeof(int)));

    size_t shm_mem_calc = thr_blc * sizeof(float) * 7;
    size_t shm_mem_mass = (red_thr_blc <= 32) ? 2 * red_thr_blc * sizeof(float) * 4 : red_thr_blc * sizeof(float) * 4;

    cudaStream_t cm_stream, cp_stream, wp_stream;
    checkCudaErrors(cudaStreamCreate(&cm_stream));
    checkCudaErrors(cudaStreamCreate(&cp_stream));
    checkCudaErrors(cudaStreamCreate(&wp_stream));

    cudaEvent_t cm_event, cp_event, wp_event;
    checkCudaErrors(cudaEventCreate(&cm_event));
    checkCudaErrors(cudaEventCreate(&cp_event));
    checkCudaErrors(cudaEventCreate(&wp_event));

    size_t records = 0;

    gettimeofday(&t1, 0);

    for (int s = 0; s < steps; s++) {
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                       FILL IN: kernels invocation (step 0)                                 //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        calculate_velocity<<< simulationGrid, thr_blc, shm_mem_calc, cp_stream >>>
            (particles_gpu[s & 1ul], particles_gpu[(s + 1) & 1ul], N, dt);
        checkCudaErrors(cudaEventRecord(cp_event, cp_stream));

        checkCudaErrors(cudaMemcpyAsync(
            particles_cpu.pos_x, particles_gpu[s & 1ul].pos_x, size, cudaMemcpyDeviceToHost, wp_stream
        ));
        checkCudaErrors(cudaMemcpyAsync(
            particles_cpu.pos_y, particles_gpu[s & 1ul].pos_y, size, cudaMemcpyDeviceToHost, wp_stream
        ));
        checkCudaErrors(cudaMemcpyAsync(
            particles_cpu.pos_z, particles_gpu[s & 1ul].pos_z, size, cudaMemcpyDeviceToHost, wp_stream
        ));
        checkCudaErrors(cudaMemcpyAsync(
            particles_cpu.vel_x, particles_gpu[s & 1ul].vel_x, size, cudaMemcpyDeviceToHost, wp_stream
        ));
        checkCudaErrors(cudaMemcpyAsync(
            particles_cpu.vel_y, particles_gpu[s & 1ul].vel_y, size, cudaMemcpyDeviceToHost, wp_stream
        ));
        checkCudaErrors(cudaMemcpyAsync(
            particles_cpu.vel_z, particles_gpu[s & 1ul].vel_z, size, cudaMemcpyDeviceToHost, wp_stream
        ));
        checkCudaErrors(cudaMemcpyAsync(
            particles_cpu.weight, particles_gpu[s & 1ul].weight, size, cudaMemcpyDeviceToHost, wp_stream
        ));

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        //                                          FILL IN: synchronization  (step 4)                                //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        checkCudaErrors(cudaMemsetAsync(comGPU, 0, sizeof(float4), cm_stream));
        compute_gpu_center_of_mass(
                particles_gpu[s & 1ul], &comGPU[0], &lock[0], N, reductionGrid, red_thr_blc, shm_mem_mass, &cm_stream
        );
        checkCudaErrors(cudaMemcpyAsync(comCPU, comGPU, sizeof(float4), cudaMemcpyDeviceToHost, cm_stream));

        if (writeFreq > 0 && (s % writeFreq == 0)) {
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //                          FILL IN: synchronization and file access logic (step 4)                       //
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////
            // Writing final values to the file
            checkCudaErrors(cudaStreamSynchronize(cm_stream));
            h5Helper.writeCom(
                comCPU[0].x / comCPU[0].w,
                comCPU[0].y / comCPU[0].w,
                comCPU[0].z / comCPU[0].w,
                comCPU[0].w, records
            );

            checkCudaErrors(cudaStreamSynchronize(wp_stream));
            h5Helper.writeParticleData(records++);
        }

//        checkCudaErrors(cudaStreamSynchronize(cp_stream));
        checkCudaErrors(cudaStreamWaitEvent(wp_stream, cp_event, 0));
        checkCudaErrors(cudaStreamWaitEvent(cm_stream, cp_event, 0));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //              FILL IN: invocation of center-of-mass kernel (step 3.1, step 3.2, step 4)                         //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    cudaDeviceSynchronize();

    checkCudaErrors(cudaMemset(comGPU, 0, sizeof(float4)));
    compute_gpu_center_of_mass(
            particles_gpu[steps & 1], &comGPU[0], &lock[0], N, reductionGrid, red_thr_blc, shm_mem_mass, &cm_stream
    );
    checkCudaErrors(cudaMemcpy(comCPU, comGPU, sizeof(float4), cudaMemcpyDeviceToHost));

    gettimeofday(&t2, 0);

    // Approximate simulation wall time
    double t = (1000000.0 * (t2.tv_sec - t1.tv_sec) + t2.tv_usec - t1.tv_usec) / 1000000.0;
    printf("Time: %f s\n", t);

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                             FILL IN: memory transfers for particle data (step 0)                               //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    checkCudaErrors(cudaMemcpy(particles_cpu.pos_x, particles_gpu[steps & 1].pos_x, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.pos_y, particles_gpu[steps & 1].pos_y, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.pos_z, particles_gpu[steps & 1].pos_z, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.vel_x, particles_gpu[steps & 1].vel_x, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.vel_y, particles_gpu[steps & 1].vel_y, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.vel_z, particles_gpu[steps & 1].vel_z, size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(particles_cpu.weight, particles_gpu[steps & 1].weight, size, cudaMemcpyDeviceToHost));

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                        FILL IN: memory transfers for center-of-mass (step 3.1, step 3.2)                       //
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
    h5Helper.writeComFinal(
        comCPU[0].x / comCPU[0].w,
        comCPU[0].y / comCPU[0].w,
        comCPU[0].z / comCPU[0].w,
        comCPU[0].w / comCPU[0].w
    );
    h5Helper.writeParticleDataFinal();

    checkCudaErrors(cudaStreamDestroy(cp_stream));
    checkCudaErrors(cudaStreamDestroy(wp_stream));
    checkCudaErrors(cudaStreamDestroy(cm_stream));

    checkCudaErrors(cudaFreeHost(particles_cpu.pos_x));
    checkCudaErrors(cudaFreeHost(particles_cpu.pos_y));
    checkCudaErrors(cudaFreeHost(particles_cpu.pos_z));
    checkCudaErrors(cudaFreeHost(particles_cpu.vel_x));
    checkCudaErrors(cudaFreeHost(particles_cpu.vel_y));
    checkCudaErrors(cudaFreeHost(particles_cpu.vel_z));
    checkCudaErrors(cudaFreeHost(particles_cpu.weight));
    checkCudaErrors(cudaFreeHost(comCPU));

    for (auto p_gpu : particles_gpu) {
        checkCudaErrors(cudaFree(p_gpu.pos_x));
        checkCudaErrors(cudaFree(p_gpu.pos_y));
        checkCudaErrors(cudaFree(p_gpu.pos_z));
        checkCudaErrors(cudaFree(p_gpu.vel_x));
        checkCudaErrors(cudaFree(p_gpu.vel_y));
        checkCudaErrors(cudaFree(p_gpu.vel_z));
        checkCudaErrors(cudaFree(p_gpu.weight));
    }
    checkCudaErrors(cudaFree(comGPU));
    checkCudaErrors(cudaFree(lock));

    return 0;
}// end of main
//----------------------------------------------------------------------------------------------------------------------

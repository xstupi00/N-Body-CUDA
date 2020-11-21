# Parallel Computations on GPU (PCG 2020)
# Assignment no. 1 (cuda)
# Login: xstupi00


N=4096
N_BIG=30720
DT=0.01f
STEPS=500
THREADS_PER_BLOCK=1024
RED_THREADS=4096
RED_THREADS_PER_BLOCK=128
WRITE_INTESITY=20

INPUT=../sampledata/sampleInput.h5
INPUT_BIG=../sampledata/30720.h5
OUTPUT=step0Output.h5
OUTPUT_BIG=stepBOutput.h5


INCLUDE=../commons
LIBS=-lhdf5

FLAGS=

.PHONY: all clean run profile

all: nbody

nbody: nbody.cu main.cu nbody.h
	nvcc ${FLAGS} -I${INCLUDE} nbody.cu main.cu ../commons/h5Helper.cpp ${LIBS} -o nbody

clean:
	rm -f *.o nbody

run:
	./nbody ${N} ${DT} ${STEPS} ${THREADS_PER_BLOCK} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT) $(OUTPUT)

run_big:
	./nbody ${N_BIG} ${DT} ${STEPS} ${THREADS_PER_BLOCK} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT_BIG) $(OUTPUT_BIG)

profile:
	nvprof \
		--devices 0 \
		--metrics flops_sp \
		--metrics flops_dp \
		--metrics flops_sp_add \
		--metrics flops_sp_mul \
		--metrics flops_sp_fma \
		--metrics flops_sp_special \
		--metrics flop_sp_efficiency \
		--metrics sm_efficiency \
		--metrics alu_fu_utilization \
		--metrics gld_transactions \
		--metrics l1_cache_global_hit_rate \
		--metrics shared_load_transactions \
	    --metrics gld_efficiency \
	    --metrics gld_requested_throughput	\
	    --metrics gst_requested_throughput \
		./nbody ${N} ${DT} 1 ${THREADS_PER_BLOCK} 0  ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT) $(OUTPUT)

profile_big:
	nvprof \
		--devices 0 \
		--metrics flops_sp \
		--metrics flops_dp \
		--metrics flops_sp_add \
		--metrics flops_sp_mul \
		--metrics flops_sp_fma \
		--metrics flops_sp_special \
		--metrics flop_sp_efficiency \
		--metrics sm_efficiency \
		--metrics alu_fu_utilization \
		--metrics gld_transactions \
		--metrics l1_cache_global_hit_rate \
		--metrics shared_load_transactions \
	    --metrics gld_efficiency \
	    --metrics gld_requested_throughput	\
	    --metrics gst_requested_throughput \
		--metrics achieved_occupancy \
		--metrics shared_efficiency \
		--metrics gld_efficiency \
		./nbody ${N_BIG} ${DT} 1 ${THREADS_PER_BLOCK} 0  ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT_BIG) $(OUTPUT_BIG)

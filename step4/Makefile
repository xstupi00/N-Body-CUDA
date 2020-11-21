# Paralelní programování na GPU (PCG 2020)
# Projekt c. 1 (cuda)
# Login: xlogin00


N=4096
N_BIG=30720
N_TEST=131072
DT=0.01f
STEPS=500
THREADS_PER_BLOCK=1024
RED_THREADS=4096
RED_THREADS_PER_BLOCK=128
WRITE_INTESITY=20

INPUT=../sampledata/sampleInput.h5
INPUT_BIG=../sampledata/30720.h5
INPUT_TEST=../sampledata/131072.h5
OUTPUT=step4Output.h5
OUTPUT_BIG=stepBOutput.h5
OUTPUT_TEST=stepTOutput.h5


INCLUDE=../commons
LIBS=-lhdf5

FLAGS=-lineinfo

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

run_test:
	./nbody ${N_TEST} ${DT} ${STEPS} ${THREADS_PER_BLOCK} ${WRITE_INTESITY} ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT_TEST) $(OUTPUT_TEST)

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
		--metrics achieved_occupancy \
		--metrics shared_efficiency \
		--metrics gld_efficiency \
		--metrics gst_efficiency \
		--metrics gst_throughput \
		--metrics gld_throughput \
		./nbody ${N} ${DT} 1 ${T} 0 ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT) $(OUTPUT)

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
	    --metrics gst_efficiency \
		--metrics gst_throughput \
		--metrics gld_throughput \
		./nbody ${N_BIG} ${DT} 1 ${THREADS_PER_BLOCK} 0  ${RED_THREADS} ${RED_THREADS_PER_BLOCK} $(INPUT_BIG) $(OUTPUT_BIG)

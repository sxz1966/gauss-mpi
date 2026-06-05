#!/bin/sh
#PBS -N hybrid_all_sizes
#PBS -e hybrid_all_sizes.e
#PBS -o hybrid_all_sizes.o
#PBS -l nodes=2:ppn=8
#PBS -l walltime=00:30:00

NODES=$(cat $PBS_NODEFILE | sort | uniq)
for node in $NODES; do
    scp master_ubss1:/home/${USER}/main_omp ${node}:/home/${USER} 1>&2
done

echo "n,mpi_procs,omp_threads,time_ms"

# 固定总线程数 = 8 (因为每个节点8核，这里用2节点共16核但只使用8线程，也可以改用1节点)
# 测试不同规模
for n in 512 1024 2048; do
    # 1 进程 × 8 线程 (纯 OpenMP)
    export OMP_NUM_THREADS=8
    /usr/local/bin/mpiexec -np 1 -machinefile $PBS_NODEFILE /home/${USER}/main_omp $n

    # 2 进程 × 4 线程
    export OMP_NUM_THREADS=4
    /usr/local/bin/mpiexec -np 2 -machinefile $PBS_NODEFILE /home/${USER}/main_omp $n

    # 4 进程 × 2 线程
    export OMP_NUM_THREADS=2
    /usr/local/bin/mpiexec -np 4 -machinefile $PBS_NODEFILE /home/${USER}/main_omp $n

    # 8 进程 × 1 线程 (纯 MPI)
    export OMP_NUM_THREADS=1
    /usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main_omp $n
done

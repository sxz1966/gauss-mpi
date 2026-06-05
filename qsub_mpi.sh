#!/bin/sh
#PBS -N gauss_mpi_all
#PBS -e test_all.e
#PBS -o test_all.o
#PBS -l nodes=2:ppn=8
#PBS -l walltime=01:00:00

NODES=$(cat $PBS_NODEFILE | sort | uniq)
for node in $NODES; do
    scp master_ubss1:/home/${USER}/mpi_gauss ${node}:/home/${USER} 1>&2
done

# 输出表头
echo "=========================================="
echo "MPI Gaussian Elimination Performance Test"
echo "=========================================="
echo "n,processes,time_ms"

# 测试不同规模和进程数
for n in 512 1024 2048; do
    for np in 1 2 4 8 16; do
        # 如果 np 超过总核心数（16）则跳过
        if [ $np -gt 16 ]; then continue; fi
        echo "Testing n=$n, np=$np" >&2
        /usr/local/bin/mpiexec -np $np -machinefile $PBS_NODEFILE /home/${USER}/mpi_gauss $n
    done
done

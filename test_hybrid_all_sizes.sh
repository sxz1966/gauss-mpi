#!/bin/sh
#PBS -N hybrid_all_sizes
#PBS -e hybrid_all_sizes.e
#PBS -o hybrid_all_sizes.o
#PBS -l nodes=1:ppn=8
#PBS -l walltime=00:30:00

NODES=$(cat $PBS_NODEFILE | sort | uniq)
for node in $NODES; do
    scp master_ubss1:/home/${USER}/main ${node}:/home/${USER} 1>&2
done

export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "n,mpi_procs,omp_threads,time_ms"

for n in 512 1024 2048; do
    # 1x8
    export OMP_NUM_THREADS=8
    /usr/local/bin/mpiexec -np 1 -machinefile $PBS_NODEFILE /home/${USER}/main $n
    # 2x4
    export OMP_NUM_THREADS=4
    /usr/local/bin/mpiexec -np 2 -machinefile $PBS_NODEFILE /home/${USER}/main $n
    # 4x2
    export OMP_NUM_THREADS=2
    /usr/local/bin/mpiexec -np 4 -machinefile $PBS_NODEFILE /home/${USER}/main $n
    # 8x1
    export OMP_NUM_THREADS=1
    /usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/main $n
done

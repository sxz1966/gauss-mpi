#!/bin/sh
#PBS -N cyclic_test
#PBS -e cyclic.e
#PBS -o cyclic.o
#PBS -l nodes=2:ppn=8
#PBS -l walltime=00:30:00

NODES=$(cat $PBS_NODEFILE | sort | uniq)
for node in $NODES; do
    scp master_ubss1:/home/${USER}/main_cyclic ${node}:/home/${USER} 1>&2
done

echo "n,mpi_procs,omp_threads,time_ms"
export OMP_NUM_THREADS=1

for n in 512 1024 2048; do
    for np in 1 2 4 8 16; do
        if [ $np -gt 16 ]; then continue; fi
        /usr/local/bin/mpiexec -np $np -machinefile $PBS_NODEFILE /home/${USER}/main_cyclic $n
    done
done

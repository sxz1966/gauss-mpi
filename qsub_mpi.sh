#!/bin/sh
#PBS -N gauss_mpi
#PBS -e test.e
#PBS -o test.o
#PBS -l nodes=1:ppn=8

NODES=$(cat $PBS_NODEFILE | sort | uniq)

for node in $NODES; do
    scp master_ubss1:/home/${USER}/gauss/mpi_gauss ${node}:/home/${USER} 1>&2
done

/usr/local/bin/mpiexec -np 8 -machinefile $PBS_NODEFILE /home/${USER}/mpi_gauss 2048
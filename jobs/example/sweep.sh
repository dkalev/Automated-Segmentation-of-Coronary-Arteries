for i in $(seq 1 1 $2); do
    sbatch $1
done

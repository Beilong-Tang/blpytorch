#!/bin/bash
# Show GPU/CPU busyness of the cluster's GPU nodes, and who is running what
# on each one. Useful for picking a quiet node before submitting a job, or
# for explaining why a job is running slower than expected (node contention).
#
# Usage: ./slurm/node_busy.sh [node ...]
#   With no args, checks all nodes in the "gpu" partition.

set -euo pipefail

if [ "$#" -gt 0 ]; then
    NODES=("$@")
else
    mapfile -t NODES < <(sinfo -N -h -p gpu -o "%N" | sort -u)
fi

printf "%-16s %-10s %-10s %-8s %-8s\n" "NODE" "GPU(alloc/tot)" "CPU(alloc/tot)" "CPULoad" "State"
printf "%-16s %-10s %-10s %-8s %-8s\n" "----" "--------------" "---------------" "-------" "-----"

for node in "${NODES[@]}"; do
    info=$(scontrol show node "$node" 2>/dev/null) || continue
    cpu_alloc=$(grep -oP 'CPUAlloc=\K[0-9]+' <<<"$info")
    cpu_tot=$(grep -oP 'CPUTot=\K[0-9]+' <<<"$info")
    cpu_load=$(grep -oP 'CPULoad=\K[0-9.]+' <<<"$info")
    state=$(grep -oP 'State=\K\S+' <<<"$info")
    gpu_alloc=$(grep -oP 'AllocTRES=\S*gres/gpu=\K[0-9]+' <<<"$info")
    gpu_tot=$(grep -oP 'Gres=gpu:[a-zA-Z0-9]+:\K[0-9]+' <<<"$info")
    printf "%-16s %-10s %-10s %-8s %-8s\n" \
        "$node" "${gpu_alloc:-0}/${gpu_tot:-?}" "${cpu_alloc:-0}/${cpu_tot:-?}" "${cpu_load:-?}" "$state"
done

echo
echo "Jobs per node:"
squeue -w "$(IFS=,; echo "${NODES[*]}")" -o "%.10i %.12P %.20j %.10u %.4t %.12M %R %b" 2>/dev/null

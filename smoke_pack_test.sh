#!/bin/bash
# Packing + performance smoke test for the shadow-model workload (train-lira.py).
# Question answered: how many trainings can share ONE A100, and what does a model cost?
#
# Phases (each EPOCHS epochs, WRN-28-2, CIFAR-10, train_size 12500):
#   clean x1, poisoned x1, poisoned x2, poisoned x3, clean x3, then one lira-inference.
# Everything goes to smoke-pack/ and does not touch the real result folders.
#
# Run from the repo root:  sbatch smoke_pack.sh
#SBATCH --job-name=mia-pack
#SBATCH --partition=a100
#SBATCH --qos=a100-30min
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pack-%j.out
#SBATCH --error=logs/pack-%j.err

set -eo pipefail

ml purge
ml Miniconda3/24.7.1-0
command -v conda >/dev/null || { echo "conda not on PATH after ml load" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda activate artifact-eval
[[ "$CONDA_DEFAULT_ENV" == artifact-eval ]] || { echo "env not active" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f cifar10-shuffle.pkl ]] || { echo "cifar10-shuffle.pkl missing: run train.py once first" >&2; exit 1; }

NET=wideresnet282
DATASET=cifar10
TRAIN_SIZE=12500
EPOCHS=3                 # epoch 0 is warm-up and is excluded from the s/epoch figure
OUT=smoke-pack
CKPT=$OUT/ckpt
mkdir -p "$OUT/logs"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

monitor_start () {       # $1 = phase name; samples GPU every 2 s
    ( while true; do
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits >> "$OUT/logs/$1.gpu"
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits >> "$OUT/logs/$1.apps"
        sleep 2
      done ) &
    MON_PID=$!
}
monitor_stop () { kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }

run_phase () {           # $1 = phase name, $2 = poison (0/1), $3... = shadow IDs to run concurrently
    local phase=$1 poison=$2; shift 2
    local snl=$(( 1 - poison )) pids=() id
    echo; echo "=== phase $phase: $# process(es), poison=$poison ==="
    monitor_start "$phase"
    local t0=$SECONDS
    for id in "$@"; do
        python train-lira.py --lr 0.1 --net_type $NET --dataset $DATASET --train_size $TRAIN_SIZE \
            --epoch $EPOCHS --modify_loss_module $poison --single_norm_layer $snl \
            --synthetic_mean 0. --synthetic_std 0.1 \
            --expID $id --total_models 128 --save_tag smoke-$phase-$id \
            --res_folder $OUT/res-$phase --ckpt_dir $CKPT \
            > "$OUT/logs/$phase.proc$id.log" 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "process $p failed in phase $phase" >&2; monitor_stop; exit 1; }; done
    echo "$(( SECONDS - t0 ))" > "$OUT/logs/$phase.wall"
    monitor_stop
}

mkdir -p "$CKPT"
run_phase clean1  0 0
run_phase poison1 1 0
run_phase poison2 1 1 2
run_phase poison3 1 3 4 5
run_phase clean3  0 1 2 3

# ---- inference cost + output sizes (one poisoned shadow) ----
echo; echo "=== inference timing ==="
t0=$SECONDS
python lira-inference.py --resume_path $CKPT/$DATASET-trainSize-$TRAIN_SIZE-smoke-poison1-0.pth.tar \
    --train_size $TRAIN_SIZE --dataset $DATASET --eval_synthetic_samples 1 \
    --res_folder $OUT/res-poison1 --expID 0 --batch_size 1024 \
    --single_norm_layer 0 --synthetic_mean 0. --synthetic_std 0.1 > "$OUT/logs/inference.log" 2>&1
INF_S=$(( SECONDS - t0 ))

# ---- summary ----
python - "$OUT" "$EPOCHS" "$INF_S" <<'EOF'
import glob, os, re, sys
out, epochs, inf_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pat = re.compile(r"Elapsed time : (\d+) hr, (\d+) min, (\d+) sec")

def steady_s_per_epoch(log):
    # "Elapsed time" is cumulative (train + test per epoch), 1 s resolution
    t = [int(h)*3600 + int(m)*60 + int(s) for h, m, s in pat.findall(open(log).read())]
    return (t[-1] - t[0]) / (len(t) - 1) if len(t) >= 2 else float("nan")

rows = []
for phase in ["clean1", "poison1", "poison2", "poison3", "clean3"]:
    logs = sorted(glob.glob(f"{out}/logs/{phase}.proc*.log"))
    if not logs: continue
    n = len(logs)
    spe = sum(steady_s_per_epoch(l) for l in logs) / n
    gpu = [tuple(map(float, l.split(","))) for l in open(f"{out}/logs/{phase}.gpu") if l.strip()]
    tail = gpu[int(len(gpu)*0.4):] or gpu          # skip startup/data loading, GPU idle there
    util = sum(u for u, _ in tail) / len(tail)
    mem_total = max(m for _, m in gpu)
    apps = [float(l.split(",")[1]) for l in open(f"{out}/logs/{phase}.apps") if "," in l]
    mem_proc = max(apps) if apps else float("nan")
    wall = int(open(f"{out}/logs/{phase}.wall").read())
    rows.append((phase, n, spe, util, mem_total, mem_proc, wall))

solo = {r[0][:-1]: r[2] for r in rows if r[1] == 1}    # 'clean' / 'poison' -> solo s/epoch
print("\n=== PACKING SUMMARY (steady state, epoch 0 excluded) ===")
print(f"{'phase':8} {'n':>2} {'s/epoch':>8} {'speedup':>8} {'GPUutil%':>9} {'GPUmem MiB':>11} {'MiB/proc':>9} "
      f"{'wall s':>7} {'GPU-h per model':>16} {'GPU-h 128 models':>17}")
for phase, n, spe, util, mem_total, mem_proc, wall in rows:
    base = solo.get(phase[:-1], float("nan"))
    speedup = n * base / spe                         # throughput relative to one process alone
    per_model = (200 * spe + inf_s) / n / 3600       # 200 epochs + inference, GPU shared by n
    print(f"{phase:8} {n:>2} {spe:>8.1f} {speedup:>8.2f} {util:>9.0f} {mem_total:>11.0f} {mem_proc:>9.0f} "
          f"{wall:>7} {per_model:>16.2f} {128*per_model:>17.1f}")
print(f"\ninference (1 poisoned shadow, incl. startup): {inf_s} s")
print("startup overhead per process = wall - epochs x s/epoch (data loading, CUDA init, warm-up)")
EOF

echo; echo "=== disk per model ==="
du -sh $CKPT/$DATASET-trainSize-$TRAIN_SIZE-smoke-poison1-0.pth.tar
du -ch $OUT/res-poison1/0_* | tail -1
echo; echo "== PACK SMOKE TEST DONE in $SECONDS s"

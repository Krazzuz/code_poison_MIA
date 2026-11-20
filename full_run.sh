#!/bin/bash
#
# Full run: WRN-28-2 on CIFAR-10, code-poisoning MIA reproduction.
#   array task 0      : clean + poisoned TARGET model (packed on one GPU), then inference
#   array task 1..128 : two LiRA SHADOW models per task (packed on one GPU), then inference
#                       odd tasks = clean shadows, even tasks = poisoned shadows
#                       -> 64 tasks x 2 = 128 shadows per arm, both arms grow at the same rate
#
#
#SBATCH --job-name=mia-full
#SBATCH --partition=a100
#SBATCH --qos=a100-6hours
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs_full_run/task-%A_%a.out
#SBATCH --error=logs_full_run/task-%A_%a.err

set -eo pipefail

ml purge
ml Miniconda3/24.7.1-0
command -v conda >/dev/null || { echo "conda not on PATH after ml load" >&2; exit 1; }
eval "$(conda shell.bash hook)"
conda activate artifact-eval
[[ "$CONDA_DEFAULT_ENV" == artifact-eval ]] || { echo "env not active" >&2; exit 1; }

cd "${SLURM_SUBMIT_DIR:-.}"
[[ -f cifar10-shuffle.pkl ]] || { echo "cifar10-shuffle.pkl missing: run train.py once first" >&2; exit 1; }
TASK=${SLURM_ARRAY_TASK_ID:?"submit as a job array, e.g. sbatch --array=0-8%3 full_run.sh"}

#config
NET=wideresnet282
DATASET=cifar10
TRAIN_SIZE=12500
EPOCHS=200
TOTAL_MODELS=128
SYN="--synthetic_mean 0. --synthetic_stdev 0.1"
OUT=results_full_run
LOGS=$OUT/logs
mkdir -p "$LOGS" "$OUT/targets"

# arm settings: name, modify_loss_module, single_norm_layer, eval_synthetic_samples
arm_flags () {   # $1 = clean|poisoned -> sets POISON SNL TAG
    if [[ $1 == clean ]]; then POISON=0; SNL=1; else POISON=1; SNL=0; fi
    TAG=$NET-$1
}

# keep a copy of the data split
[[ -f $OUT/cifar10-shuffle.pkl ]] || cp cifar10-shuffle.pkl "$OUT/"
cmp -s cifar10-shuffle.pkl "$OUT/cifar10-shuffle.pkl" \
    || { echo "cifar10-shuffle.pkl changed since the run started: data split differs!" >&2; exit 1; }

echo "== task $TASK on $(hostname), job $SLURM_JOB_ID, $(date -Is)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
pip freeze > "$LOGS/pip-freeze-$SLURM_JOB_ID.txt"

# GPU monitor (every 10 s)
( while true; do
    echo "$(date +%s),$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits)"
    sleep 10
  done ) > "$LOGS/task-$TASK.gpu.csv" &
MON_PID=$!
trap 'kill $MON_PID 2>/dev/null || true' EXIT

# timing log: task,model,step,seconds
timed () {   # $1 = model label, $2 = step, rest = command
    local label=$1 step=$2; shift 2
    local t0=$SECONDS
    "$@"
    echo "$TASK,$label,$step,$(( SECONDS - t0 ))" >> "$LOGS/timing.csv"
}

# target models
target_one () {   # $1 = clean|poisoned
    arm_flags "$1"
    local tag=$TAG-target
    local ckpt=checkpoint/$DATASET-trainSize-$TRAIN_SIZE-$tag.pth.tar
    local res=$OUT/targets/res-$1
    if [[ -f $res/0_shadow_sample_logit.npy && -f $OUT/targets/$tag.pth.tar ]]; then
        echo "target $1: done, skipping"; return 0
    fi
    timed "target-$1" train python train.py --dataset $DATASET --lr 0.1 --net_type $NET \
        --train_size $TRAIN_SIZE --epoch $EPOCHS --modify_loss_module $POISON \
        --single_norm_layer $SNL $SYN --save_tag $tag
    cp "$ckpt" "$OUT/targets/$tag.pth.tar"          # keep for the demo
    timed "target-$1" inference python lira-inference.py --resume_path "$ckpt" \
        --train_size $TRAIN_SIZE --dataset $DATASET --eval_synthetic_samples $POISON \
        --single_norm_layer $SNL $SYN --res_folder "$res" --target_model 1 --batch_size 1024
}

# shadow models
shadow_one () {   # $1 = clean|poisoned, $2 = shadow ID
    arm_flags "$1"
    local id=$2
    local res=$OUT/shadows-$1/res
    local ckdir=$OUT/shadows-$1/ckpt
    local ckpt=$ckdir/$DATASET-trainSize-$TRAIN_SIZE-$TAG-$id.pth.tar
    if [[ -f $res/${id}_shadow_sample_logit.npy && -f $res/${id}_testSet_logit.npy ]]; then
        echo "shadow $1 $id: done, skipping"; return 0
    fi
    timed "shadow-$1-$id" train python train-lira.py --lr 0.1 --net_type $NET --dataset $DATASET \
        --train_size $TRAIN_SIZE --epoch $EPOCHS --modify_loss_module $POISON \
        --single_norm_layer $SNL $SYN --expID $id --total_models $TOTAL_MODELS \
        --save_tag $TAG-$id --res_folder "$res" --ckpt_dir "$ckdir"
    timed "shadow-$1-$id" inference python lira-inference.py --resume_path "$ckpt" \
        --train_size $TRAIN_SIZE --dataset $DATASET --eval_synthetic_samples $POISON \
        --single_norm_layer $SNL $SYN --res_folder "$res" --expID $id --batch_size 1024
}

run_pair () {   # runs two background jobs, fails the task if either fails
    local log1=$1 log2=$2; shift 2
    "${JOB1[@]}" >> "$log1" 2>&1 & local p1=$!
    "${JOB2[@]}" >> "$log2" 2>&1 & local p2=$!
    local rc=0
    wait $p1 || { echo "FAILED: ${JOB1[*]} (see $log1)" >&2; rc=1; }
    wait $p2 || { echo "FAILED: ${JOB2[*]} (see $log2)" >&2; rc=1; }
    return $rc
}

if (( TASK == 0 )); then
    JOB1=(target_one clean); JOB2=(target_one poisoned)
    run_pair "$LOGS/target-clean.log" "$LOGS/target-poisoned.log"
elif (( TASK >= 1 && TASK <= 128 )); then
    k=$(( TASK - 1 ))
    if (( k % 2 == 0 )); then ARM=clean; else ARM=poisoned; fi
    pair=$(( k / 2 ))
    a=$(( 2 * pair )); b=$(( 2 * pair + 1 ))  # shadow IDs 0..127
    mkdir -p "$OUT/shadows-$ARM/res" "$OUT/shadows-$ARM/ckpt"
    JOB1=(shadow_one $ARM $a); JOB2=(shadow_one $ARM $b)
    run_pair "$LOGS/shadow-$ARM-$a.log" "$LOGS/shadow-$ARM-$b.log"
else
    echo "task $TASK out of range 0..128" >&2; exit 1
fi

echo "== task $TASK finished in $SECONDS s, $(date -Is)"
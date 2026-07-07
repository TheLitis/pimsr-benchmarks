#!/usr/bin/env bash
# PIMSR v3 (TE+TM) post-processing driver.
#
# Run after the "PIMSR 2D training" run triggered by dataset run finishes:
#   bash scripts/run_v3_postprocess.sh <TRAIN_RUN_ID> <DATASET_RUN_ID>
#
# Requires: GITHUB_TOKEN env var, pimsr venv at $VENV (default
# /vercel/share/pimsr-venv), repos checked out next to this script's repo.
set -euo pipefail

TRAIN_RUN_ID="${1:?usage: run_v3_postprocess.sh <train_run_id> <dataset_run_id>}"
DATASET_RUN_ID="${2:?usage: run_v3_postprocess.sh <train_run_id> <dataset_run_id>}"
TOKEN="${GITHUB_TOKEN:?set GITHUB_TOKEN}"

VENV="${VENV:-/vercel/share/pimsr-venv}"
DATA="${DATA:-/vercel/share/pimsr-data/v3}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
EMTF="$REPO_DIR/data/emtf"
PY="$VENV/bin/python"

api() {
  curl -sL -H "Authorization: token $TOKEN" \
    -H "Accept: application/vnd.github+json" "$@"
}

mkdir -p "$DATA"
cd "$DATA"

echo "== 1/6 download artifacts =="
for RUN in "$TRAIN_RUN_ID" "$DATASET_RUN_ID"; do
  api "https://api.github.com/repos/TheLitis/Runner/actions/runs/$RUN/artifacts" \
    | "$PY" -c "
import json, sys
for a in json.load(sys.stdin)['artifacts']:
    print(a['id'], a['name'])
" | while read -r AID NAME; do
      echo "  artifact $NAME ($AID)"
      api "https://api.github.com/repos/TheLitis/Runner/actions/artifacts/$AID/zip" -o "$NAME.zip"
      unzip -o -q "$NAME.zip" -d "$NAME" && rm "$NAME.zip"
    done
done

CKPT="$(find "$DATA" -name best2d.pt | head -1)"
TEST_H5="$(find "$DATA" -name 'ds2d_test.h5' | head -1)"
echo "ckpt: $CKPT"
echo "test: $TEST_H5"

echo "== 2/6 synthetic + real benchmark (pretrained) =="
"$PY" "$REPO_DIR/scripts/run_2d_bench.py" \
  --checkpoint "$CKPT" --test-h5 "$TEST_H5" --emtf-dir "$EMTF" \
  --out-dir "$REPO_DIR/results/v3/bench_pre" --n 500

echo "== 3/6 Yellowstone fine-tune (champion recipe aw=3/600) =="
"$PY" -m pimsr_inversion.finetune2d \
  --checkpoint "$CKPT" --emtf-dir "$EMTF" --data-h5 "$TEST_H5" \
  --out "$DATA/best2d_v3_ft_YS.pt" --steps 600 --anchor-weight 3

echo "== 4/6 joint multi-profile fine-tune (regional model) =="
"$PY" -m pimsr_inversion.finetune2d \
  --checkpoint "$CKPT" --emtf-dir "$EMTF" --data-h5 "$TEST_H5" \
  --out "$DATA/best2d_v3_ft_joint.pt" --steps 600 --anchor-weight 3 \
  --profiles G,H-YS,I,J,K

echo "== 5/6 benchmark fine-tuned variants =="
"$PY" "$REPO_DIR/scripts/run_2d_bench.py" \
  --checkpoint "$DATA/best2d_v3_ft_YS.pt" --test-h5 "$TEST_H5" \
  --emtf-dir "$EMTF" --out-dir "$REPO_DIR/results/v3/bench_ft_ys" --n 500
"$PY" "$REPO_DIR/scripts/run_2d_bench.py" \
  --checkpoint "$DATA/best2d_v3_ft_joint.pt" --test-h5 "$TEST_H5" \
  --emtf-dir "$EMTF" --out-dir "$REPO_DIR/results/v3/bench_ft_joint" --n 500

echo "== 6/6 unified leaderboard with v3 rows =="
"$PY" "$REPO_DIR/scripts/run_unified_leaderboard.py" \
  --test-h5 "$TEST_H5" --emtf-dir "$EMTF" \
  --ckpt-1d "${CKPT_1D:-/vercel/share/pimsr-data/ckpts/1d/best.pt}" \
  --ckpt-10k "${CKPT_10K:-/vercel/share/pimsr-data/ckpts/10k/best2d.pt}" \
  --ckpt-10k-ft "${CKPT_10K_FT:-/vercel/share/pimsr-data/leaderboard/ft/best2d_ft_YS_10k.pt}" \
  --ckpt-60k "${CKPT_60K:-/vercel/share/pimsr-data/60k/best2d.pt}" \
  --ckpt-60k-ft "${CKPT_60K_FT:-/vercel/share/pimsr-data/mpft/best2d_ft_YS.pt}" \
  --ckpt-v3 "$CKPT" --ckpt-v3-ft "$DATA/best2d_v3_ft_YS.pt" \
  --skip-gn \
  --out "$REPO_DIR/results/v3/unified_v3.json" || \
  echo "WARN: leaderboard needs 1d/10k/60k ckpts; set CKPT_1D/CKPT_10K/CKPT_10K_FT/CKPT_60K/CKPT_60K_FT"

echo "== done. Key numbers: =="
for f in "$REPO_DIR"/results/v3/bench_pre/*.json \
         "$REPO_DIR"/results/v3/bench_ft_ys/*.json \
         "$REPO_DIR"/results/v3/bench_ft_joint/*.json; do
  [ -f "$f" ] && echo "--- $f" && "$PY" -m json.tool "$f" | head -20
done

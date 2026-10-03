#!/usr/bin/env bash
# Reproduces the synthetic-intersection validation (stitching ablation).
set -e
python src/synth.py --out data/synth && cp data/synth.yaml configs/synth.yaml
for m in stitch nostitch; do
  F=""; [ $m = nostitch ] && F="--no-stitch"
  python -W ignore src/pipeline.py --video data/synth.mp4 --config configs/synth.yaml --detector gt \
      --gt data/synth_gt.json --out output/synth_$m --no-video $F > /dev/null
  echo "=== $m"; python src/evaluate.py --gt data/synth_gt.json --run output/synth_$m | head -14
done

# Reproduces the synthetic-intersection validation (stitching ablation).
# Run from the repo root with the venv active.
python src/synth.py --out data/synth
Copy-Item data/synth.yaml configs/synth.yaml -Force
foreach ($m in 'stitch', 'nostitch') {
    $extra = @()
    if ($m -eq 'nostitch') { $extra = @('--no-stitch') }
    python -W ignore src/pipeline.py --video data/synth.mp4 --config configs/synth.yaml --detector gt --gt data/synth_gt.json --out "output/synth_$m" --no-video @extra | Out-Null
    "=== $m"
    python src/evaluate.py --gt data/synth_gt.json --run "output/synth_$m" | Select-Object -First 14
}

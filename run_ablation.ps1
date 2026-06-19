$ErrorActionPreference = "SilentlyContinue"
$BASE = "python simulation/kg/train_manual.py --batch-size 4350 --num-negatives 4 --grad-accum-steps 2 --epochs 50 --eval --patience 10 --eval-batch-size 512 --eval-every-epochs 3"

foreach ($abl in @("no_pid", "no_type", "no_modality")) {
    $log = "ckpts/ablation/$abl.log"
    $ckpt = "ckpts/ablation/$abl"
    Write-Output "=== $abl ==="
    Invoke-Expression "& $BASE --ablation $abl --checkpoint-dir $ckpt 2>&1 | Out-File $log"
    $ec = $LASTEXITCODE
    Write-Output "exit: $ec"
}

Write-Output "=== ALL DONE ==="

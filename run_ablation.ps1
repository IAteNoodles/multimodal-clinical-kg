$BASE = "python simulation/kg/train_manual.py --batch-size 4350 --num-negatives 4 --grad-accum-steps 2 --epochs 50 --eval --patience 10 --eval-batch-size 512 --eval-every-epochs 3 --max-eval-triples 5000"

$configs = @(
    @{ Name = "full"; Ablation = "" },
    @{ Name = "no_pid"; Ablation = "--ablation no_pid" },
    @{ Name = "no_type"; Ablation = "--ablation no_type" },
    @{ Name = "no_modality"; Ablation = "--ablation no_modality" }
)

foreach ($seed in @(42, 123, 456)) {
    foreach ($cfg in $configs) {
        $name = "$($cfg.Name)_seed$seed"
        $log = "ckpts/ablation/$name.log"
        $ckpt = "ckpts/ablation/$name"
        Write-Output "=== $name ==="
        $cmd = "& $BASE $($cfg.Ablation) --seed $seed --checkpoint-dir $ckpt"
        Invoke-Expression "$cmd 2>&1 | Out-File $log"
        $ec = $LASTEXITCODE
        Write-Output "exit: $ec"
        if ($ec -ne 0) {
            Write-Error "Run failed with exit code $ec"
            exit $ec
        }
    }
}

Write-Output "=== ALL DONE ==="
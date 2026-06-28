$models = @(
    @{ Name = "transE";       BatchSize = 38000 },
    @{ Name = "complex";      BatchSize = 18000 },
    @{ Name = "multimodal_cascade"; BatchSize = 3000 }
)

$configs = @(
    @{ Name = "full"; Ablation = "" },
    @{ Name = "no_pid"; Ablation = "--ablation no_pid" },
    @{ Name = "no_type"; Ablation = "--ablation no_type" },
    @{ Name = "no_modality"; Ablation = "--ablation no_modality" }
)

if (-not (Test-Path "ckpts/ablation")) { New-Item -ItemType Directory -Path "ckpts/ablation" | Out-Null }

foreach ($m in $models) {
    $BASE = "python simulation/kg/train_manual.py --model $($m.Name) --batch-size $($m.BatchSize) --num-negatives 4 --grad-accum-steps 2 --epochs 50 --eval --patience 10 --eval-batch-size 512 --eval-every-epochs 3 --max-eval-triples 5000"

    if ($m.Name -eq "multimodal_cascade") {
        foreach ($seed in @(42, 123, 456)) {
            foreach ($cfg in $configs) {
                $name = "$($m.Name)_$($cfg.Name)_seed$seed"
                $ckpt = "ckpts/ablation/$name"
                $log = "ckpts/ablation/$name.log"
                if (Test-Path "$ckpt/meta.json") {
                    Write-Output "SKIP $name (already done)"
                } elseif (Test-Path "$ckpt") {
                    Write-Output "RESUME $name (incomplete)"
                    $cmd = "& $BASE $($cfg.Ablation) --seed $seed --checkpoint-dir $ckpt --resume"
                    Invoke-Expression "$cmd 2>&1 | Out-File $log"
                    $ec = $LASTEXITCODE
                    Write-Output "exit: $ec"
                    if ($ec -ne 0) { Write-Error "Run failed with exit code $ec"; exit $ec }
                } else {
                    Write-Output "=== $name ==="
                    $cmd = "& $BASE $($cfg.Ablation) --seed $seed --checkpoint-dir $ckpt"
                    Invoke-Expression "$cmd 2>&1 | Out-File $log"
                    $ec = $LASTEXITCODE
                    Write-Output "exit: $ec"
                    if ($ec -ne 0) { Write-Error "Run failed with exit code $ec"; exit $ec }
                }
            }
        }
    } else {
        foreach ($seed in @(42, 123, 456)) {
            $name = "$($m.Name)_seed$seed"
            $ckpt = "ckpts/ablation/$name"
            $log = "ckpts/ablation/$name.log"
            if (Test-Path "$ckpt/meta.json") {
                Write-Output "SKIP $name (already done)"
            } elseif (Test-Path "$ckpt") {
                Write-Output "RESUME $name (incomplete)"
                $cmd = "& $BASE --seed $seed --checkpoint-dir $ckpt --resume"
                Invoke-Expression "$cmd 2>&1 | Out-File $log"
                $ec = $LASTEXITCODE
                Write-Output "exit: $ec"
                if ($ec -ne 0) { Write-Error "Run failed with exit code $ec"; exit $ec }
            } else {
                Write-Output "=== $name ==="
                $cmd = "& $BASE --seed $seed --checkpoint-dir $ckpt"
                Invoke-Expression "$cmd 2>&1 | Out-File $log"
                $ec = $LASTEXITCODE
                Write-Output "exit: $ec"
                if ($ec -ne 0) { Write-Error "Run failed with exit code $ec"; exit $ec }
            }
        }
    }
}

Write-Output "=== ALL DONE ==="

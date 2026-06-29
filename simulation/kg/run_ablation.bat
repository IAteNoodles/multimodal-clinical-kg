@echo off
set ABLATION=%1
set CKPT_DIR=ckpts\cascade_ablation_%ABLATION%
set LOG_FILE=ckpts\cascade_ablation_%ABLATION%_log.txt

if exist "%CKPT_DIR%\meta.json" (
    echo Resuming %ABLATION% from existing checkpoint...
    set RESUME=--resume
) else (
    echo Starting fresh %ABLATION% run...
    set RESUME=
)

start /B python.exe simulation/kg/train_manual.py ^
    --model multimodal_cascade --modalities text --ablation %ABLATION% ^
    --embed-dim 256 --batch-size 3000 --grad-accum-steps 1 ^
    --epochs 51 --lr 3e-4 --temperature 0.2 --label-smoothing 0.05 --n3-weight 0.01 ^
    --checkpoint-dir %CKPT_DIR% --keep-last-n 0 ^
    --eval --eval-every-epochs 3 --patience 10 --eval-batch-size 1024 ^
    %RESUME% > "%LOG_FILE%" 2>&1

echo Started %ABLATION% ablation (PID=%ERRORLEVEL%, see %LOG_FILE%)

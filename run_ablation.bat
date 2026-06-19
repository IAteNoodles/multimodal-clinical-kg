@echo off
set BASE=python simulation/kg/train_manual.py --batch-size 4350 --num-negatives 4 --grad-accum-steps 2 --epochs 50 --eval --patience 10 --eval-batch-size 512 --eval-every-epochs 3

if not exist ckpts\ablation mkdir ckpts\ablation

echo ====== no_pid ======
%BASE% --ablation no_pid --checkpoint-dir ckpts\ablation\no_pid > ckpts\ablation\no_pid.log 2>&1
if %ERRORLEVEL% NEQ 0 echo no_pid FAILED

echo ====== no_type ======
%BASE% --ablation no_type --checkpoint-dir ckpts\ablation\no_type > ckpts\ablation\no_type.log 2>&1
if %ERRORLEVEL% NEQ 0 echo no_type FAILED

echo ====== no_modality ======
%BASE% --ablation no_modality --checkpoint-dir ckpts\ablation\no_modality > ckpts\ablation\no_modality.log 2>&1
if %ERRORLEVEL% NEQ 0 echo no_modality FAILED

echo ====== ALL DONE ======

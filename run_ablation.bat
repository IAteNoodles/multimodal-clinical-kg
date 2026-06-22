@echo off
setlocal enabledelayedexpansion
set BASE=python simulation/kg/train_manual.py --batch-size 4350 --num-negatives 4 --grad-accum-steps 2 --epochs 50 --eval --patience 10 --eval-batch-size 512 --eval-every-epochs 3

if not exist ckpts\ablation mkdir ckpts\ablation

for %%S in (42 123 456) do (
    echo ====== full seed %%S ======
    %BASE% --seed %%S --checkpoint-dir ckpts\ablation\full_seed%%S > ckpts\ablation\full_seed%%S.log 2>&1
    if !ERRORLEVEL! NEQ 0 echo full seed %%S FAILED

    echo ====== no_pid seed %%S ======
    %BASE% --ablation no_pid --seed %%S --checkpoint-dir ckpts\ablation\no_pid_seed%%S > ckpts\ablation\no_pid_seed%%S.log 2>&1
    if !ERRORLEVEL! NEQ 0 echo no_pid seed %%S FAILED

    echo ====== no_type seed %%S ======
    %BASE% --ablation no_type --seed %%S --checkpoint-dir ckpts\ablation\no_type_seed%%S > ckpts\ablation\no_type_seed%%S.log 2>&1
    if !ERRORLEVEL! NEQ 0 echo no_type seed %%S FAILED

    echo ====== no_modality seed %%S ======
    %BASE% --ablation no_modality --seed %%S --checkpoint-dir ckpts\ablation\no_modality_seed%%S > ckpts\ablation\no_modality_seed%%S.log 2>&1
    if !ERRORLEVEL! NEQ 0 echo no_modality seed %%S FAILED
)

echo ====== ALL DONE ======
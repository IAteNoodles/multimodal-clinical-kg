@echo off
setlocal enabledelayedexpansion

set MASTER_LOG=ckpts\ablation\ablation_master.log
set BASE=python simulation/kg/train_manual.py --num-negatives 4 --grad-accum-steps 2 --epochs 51 --eval --patience 10 --eval-batch-size 512 --eval-every-epochs 3 --max-eval-triples 5000

if not exist ckpts\ablation mkdir ckpts\ablation
echo === started %date% %time% === > "%MASTER_LOG%"

for %%M in (transE complex multimodal_cascade) do (
    if "%%M"=="transE" set BS=38000
    if "%%M"=="complex" set BS=18000
    if "%%M"=="multimodal_cascade" set BS=3000

    if "%%M"=="multimodal_cascade" (
        for %%S in (42 123 456) do (
            for %%A in (full no_pid no_type no_modality) do (
                set CKPT=ckpts\ablation\%%M_%%A_seed%%S
                set ABL_FLAG=--ablation %%A
                if "%%A"=="full" set ABL_FLAG=
                if exist "!CKPT!\meta.json" (
                    echo SKIP %%M %%A seed %%S (already done) >> "%MASTER_LOG%"
                )
                if not exist "!CKPT!\meta.json" if exist "!CKPT!" (
                    echo RESUME %%M %%A seed %%S (incomplete) >> "%MASTER_LOG%"
                    %BASE% --model %%M --batch-size !BS! !ABL_FLAG! --seed %%S --checkpoint-dir !CKPT! --resume >> "!CKPT!.log" 2>&1
                    if !ERRORLEVEL! NEQ 0 echo %%M %%A seed %%S FAILED >> "%MASTER_LOG%"
                )
                if not exist "!CKPT!\meta.json" if not exist "!CKPT!" (
                    echo ====== %%M %%A seed %%S ====== >> "%MASTER_LOG%"
                    %BASE% --model %%M --batch-size !BS! !ABL_FLAG! --seed %%S --checkpoint-dir !CKPT! >> "!CKPT!.log" 2>&1
                    if !ERRORLEVEL! NEQ 0 echo %%M %%A seed %%S FAILED >> "%MASTER_LOG%"
                )
            )
        )
    ) else (
        for %%S in (42 123 456) do (
            set CKPT=ckpts\ablation\%%M_seed%%S
            if exist "!CKPT!\meta.json" (
                echo SKIP %%M seed %%S (already done) >> "%MASTER_LOG%"
            )
            if not exist "!CKPT!\meta.json" if exist "!CKPT!" (
                echo RESUME %%M seed %%S (incomplete) >> "%MASTER_LOG%"
                %BASE% --model %%M --batch-size !BS! --seed %%S --checkpoint-dir !CKPT! --resume >> "!CKPT!.log" 2>&1
                if !ERRORLEVEL! NEQ 0 echo %%M seed %%S FAILED >> "%MASTER_LOG%"
            )
            if not exist "!CKPT!\meta.json" if not exist "!CKPT!" (
                echo ====== %%M seed %%S ====== >> "%MASTER_LOG%"
                %BASE% --model %%M --batch-size !BS! --seed %%S --checkpoint-dir !CKPT! >> "!CKPT!.log" 2>&1
                if !ERRORLEVEL! NEQ 0 echo %%M seed %%S FAILED >> "%MASTER_LOG%"
            )
        )
    )
)

echo === finished %date% %time% === >> "%MASTER_LOG%"
echo ====== ALL DONE ======

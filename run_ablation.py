"""Ablation runner. Run: python run_ablation.py"""
import subprocess, sys, time
from pathlib import Path

BASE = [
    sys.executable, "simulation/kg/train_manual.py",
    "--num-negatives", "4", "--grad-accum-steps", "2",
    "--epochs", "50", "--eval", "--patience", "10",
    "--eval-batch-size", "512", "--eval-every-epochs", "3",
    "--max-eval-triples", "5000",
    "--max-test-triples", "5000",
    "--eval-chunk-size", "512",
    "--keep-last-n", "-1",
]

MODELS = [
    ("transE", 38000, False),
    ("complex", 1024, False),
    ("multimodal_cascade", 512, True),
]

ABLATIONS = ["full", "no_pid", "no_type", "no_modality"]
SEEDS = [42, 123, 456]
CKPT_ROOT = Path("ckpts/ablation")
CKPT_ROOT.mkdir(parents=True, exist_ok=True)

master_log = CKPT_ROOT / "ablation_master.log"

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    with open(master_log, "a") as f:
        f.write(line + "\n")

log(f"=== started ===")

for model, bs, has_ablations in MODELS:
    if has_ablations:
        for ablation in ABLATIONS:
            for seed in SEEDS:
                name = f"{model}_{ablation}_seed{seed}"
                ckpt = CKPT_ROOT / name
                logfile = CKPT_ROOT / f"{name}.log"

                if (ckpt / "meta.json").exists():
                    log(f"SKIP {name} (already done)")
                    continue

                cmd = BASE + [
                    "--model", model, "--batch-size", str(bs),
                    "--seed", str(seed), "--checkpoint-dir", str(ckpt),
                ]
                if ablation != "full":
                    cmd += ["--ablation", ablation]

                if ckpt.exists():
                    cmd.append("--resume")
                    log(f"RESUME {name}")
                else:
                    log(f"START {name}")

                with open(logfile, "a") as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

                if result.returncode != 0:
                    log(f"FAIL {name} (exit={result.returncode})")
                else:
                    log(f"DONE {name}")
    else:
        for seed in SEEDS:
            name = f"{model}_seed{seed}"
            ckpt = CKPT_ROOT / name
            logfile = CKPT_ROOT / f"{name}.log"

            if (ckpt / "meta.json").exists():
                log(f"SKIP {name} (already done)")
                continue

            cmd = BASE + [
                "--model", model, "--batch-size", str(bs),
                "--seed", str(seed), "--checkpoint-dir", str(ckpt),
            ]

            if ckpt.exists():
                cmd.append("--resume")
                log(f"RESUME {name}")
            else:
                log(f"START {name}")

            with open(logfile, "a") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

            if result.returncode != 0:
                log(f"FAIL {name} (exit={result.returncode})")
            else:
                log(f"DONE {name}")

log(f"=== finished ===")

"""Ablation runner. Run: python run_ablation.py"""
import subprocess, sys, time, json
from pathlib import Path

BASE = [
    sys.executable, "simulation/kg/train_manual.py",
    "--num-negatives", "4", "--grad-accum-steps", "2",
    "--epochs", "51", "--eval", "--patience", "0",
    "--eval-batch-size", "512", "--eval-every-epochs", "3",
    "--max-eval-triples", "5000",
    "--max-test-triples", "5000",
    "--eval-chunk-size", "512",
    "--keep-last-n", "-1",
]

TARGET_EPOCHS = int(BASE[BASE.index("--epochs") + 1])

MODELS = [
    ("transE", 38000, False, {"--lr": "3e-4", "--weight-decay": "1e-3", "--n3-weight": "0.0"}),
    ("complex", 1024, False, {"--lr": "1e-3", "--weight-decay": "0", "--n3-weight": "0.0"}),
    ("multimodal_cascade", 512, True, {"--lr": "3e-4", "--weight-decay": "1e-5", "--n3-weight": "0.01"}),
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

for model, bs, has_ablations, overrides in MODELS:
    if has_ablations:
        for ablation in ABLATIONS:
            for seed in SEEDS:
                name = f"{model}_{ablation}_seed{seed}"
                ckpt = CKPT_ROOT / name
                logfile = CKPT_ROOT / f"{name}.log"

                meta_path = ckpt / "meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    ep_done = meta.get("ep", 0)
                    if ep_done >= TARGET_EPOCHS:
                        log(f"SKIP {name} (ep={ep_done} >= {TARGET_EPOCHS})")
                        continue
                    log(f"RESUME {name} (ep={ep_done}/{TARGET_EPOCHS})")

                cmd = BASE + [
                    "--model", model, "--batch-size", str(bs),
                    "--seed", str(seed), "--checkpoint-dir", str(ckpt),
                ]
                for k, v in overrides.items():
                    cmd += [k, v]
                if ablation != "full":
                    cmd += ["--ablation", ablation]

                if ckpt.exists() and meta_path.exists():
                    cmd.append("--resume")
                elif ckpt.exists():
                    log(f"WARN {name}: dir exists but no meta.json, starting fresh")
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

            meta_path = ckpt / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                ep_done = meta.get("ep", 0)
                if ep_done >= TARGET_EPOCHS:
                    log(f"SKIP {name} (ep={ep_done} >= {TARGET_EPOCHS})")
                    continue
                log(f"RESUME {name} (ep={ep_done}/{TARGET_EPOCHS})")

            cmd = BASE + [
                "--model", model, "--batch-size", str(bs),
                "--seed", str(seed), "--checkpoint-dir", str(ckpt),
            ]
            for k, v in overrides.items():
                cmd += [k, v]

            if ckpt.exists() and meta_path.exists():
                cmd.append("--resume")
            elif ckpt.exists():
                log(f"WARN {name}: dir exists but no meta.json, starting fresh")
            else:
                log(f"START {name}")

            with open(logfile, "a") as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

            if result.returncode != 0:
                log(f"FAIL {name} (exit={result.returncode})")
            else:
                log(f"DONE {name}")

log(f"=== finished ===")

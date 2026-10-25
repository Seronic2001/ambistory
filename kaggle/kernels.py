#!/usr/bin/env python
"""Kaggle kernels for the E14 LLM judges and the E13 deploy run (the ablations have ablations.py).

Each kernel embeds the current script from this folder, so regenerate before pushing:

    python kernels.py --dir kernels/                      # one folder per kernel
    kaggle kernels push -p kernels/e14-llm-judge-32b --accelerator NvidiaTeslaT4
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OWNER = "shubhadeepmandal"
DATASETS = [f"{OWNER}/ambistory-e12", f"{OWNER}/inlp-project"]

JUDGE = '''"""E14 LLM judge, {model}, vLLM tensor parallel over 2x T4."""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
# vllm brings its own torch build; the preinstalled torchaudio no longer matches it and transformers imports it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], check=True)
open("/tmp/llm_judge.py", "w", encoding="utf8").write({src!r})
subprocess.run([sys.executable, "/tmp/llm_judge.py", "--backend", "vllm", "--tp", "2", "--model", "{model}",
                "--out", "/kaggle/working/e14"], check=True)
'''

DEPLOY = '''"""E13 deploy: DeBERTa-v3-large cross-encoder fitted on all of train+dev, 2 seeds (one per T4), saved for app.py."""
import glob, os, shutil, subprocess, sys

src = os.path.dirname(glob.glob("/kaggle/input/**/e12_train.py", recursive=True)[0])
code = "/tmp/code"
for d in ("ambistory", "kaggle"):
    os.makedirs(os.path.join(code, d), exist_ok=True)
for path in glob.glob(os.path.join(src, "ambistory__*.py")):
    shutil.copy(path, os.path.join(code, "ambistory", os.path.basename(path)[len("ambistory__"):]))
shutil.copy(os.path.join(src, "e12_train.py"), os.path.join(code, "kaggle", "e12_train.py"))
open(os.path.join(code, "kaggle", "e13_train.py"), "w", encoding="utf8").write({src!r})
gold = [p for p in glob.glob("/kaggle/input/**/test.json", recursive=True) if "inlp-project" in p]
cmd = [sys.executable, os.path.join(code, "kaggle", "e13_train.py"), "--data", src, "--out", "/kaggle/working/e13_deploy",
       "--deploy", "--epochs", "6", "--seeds", "42,43"] + (["--test_gold", gold[0]] if gold else [])
print(" ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
'''

# slug, template, script it embeds, template arguments
KERNELS = [
    ("e14-llm-judge-14b", JUDGE, "llm_judge.py", {"model": "Qwen/Qwen2.5-14B-Instruct-AWQ"}),
    ("e14-llm-judge-32b", JUDGE, "llm_judge.py", {"model": "Qwen/Qwen2.5-32B-Instruct-AWQ"}),
    ("e13-deploy", DEPLOY, "e13_train.py", {}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="kernels")
    cfg = ap.parse_args()
    for slug, template, script, args in KERNELS:
        d = os.path.join(cfg.dir, slug)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(HERE, script), encoding="utf8") as f:
            src = f.read()
        with open(os.path.join(d, "run.py"), "w", encoding="utf8") as f:
            f.write(template.format(src=src, **args))
        meta = {"id": f"{OWNER}/ambistory-{slug}", "title": f"ambistory {slug.replace('-', ' ')}", "code_file": "run.py",
                "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
                "enable_internet": True, "dataset_sources": DATASETS, "competition_sources": [], "kernel_sources": []}
        with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
    print(f"generated {len(KERNELS)} kernels in {cfg.dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train from a configs/*.yaml (exact paper commands)."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import yaml
ROOT = Path(__file__).resolve().parents[1]

def build_command(config_path: Path) -> list[str]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    command = [sys.executable, str(ROOT / config["script"])]
    for key, value in config["args"].items():
        if value is True:
            command.append(f"--{key}")
        elif value is False or value is None:
            continue
        else:
            command += [f"--{key}", str(value)]
    return command

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, help="e.g. configs/vit_residual_attn_odir7_v2_518.yaml")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = args.config if args.config.is_absolute() else ROOT / args.config
    cmd = build_command(cfg)
    print(" ".join(cmd), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(cmd, cwd=ROOT))

if __name__ == "__main__":
    main()

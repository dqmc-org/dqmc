#!/usr/bin/env python3
import os
import subprocess
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob

# Configuration
MAX_WORKERS = os.getenv("MAX_WORKERS", 4)
CONFIG_FILE = "./example/config.cfg"
OUTPUT_DIR = "./example/output_parallel"
EXE = "./build/main"
OBSERVABLE = "local_spin_corr"

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_BETAS = 4
BETA_MIN, BETA_MAX = 0.01, 20.0
betas = (1 / (1 - np.linspace(0, 1, N_BETAS+1)))
betas.resize(N_BETAS)
betas = BETA_MIN + BETA_MAX * (betas - np.min(betas))/np.ptp(betas)

N_ONSITE_U_POINTS = 8
ONSITE_U_MIN, ONSITE_U_MAX = 0.5, 12.0
onsite_Us = np.linspace(ONSITE_U_MIN, ONSITE_U_MAX, N_ONSITE_U_POINTS)

def run_simulation(beta, onsite_U, beta_id, u_id):
    seed = random.randint(1, 2**31 - 1)  # 32-bit random seed
    output_path = os.path.join(OUTPUT_DIR, f"beta_{beta_id}_{u_id}")
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        EXE,
        "--config", CONFIG_FILE,
        "--output", output_path,
        "--model.onsite_u", str(onsite_U),
        "--mc.beta", str(beta),
        "--mc.time_size", str(max(10, (160/8)*beta)),
        "--seed", str(seed)
    ]

    print(f"Running: beta={beta:.3f}, onsite_u={onsite_U:.3f}, seed={seed}, output={output_path}")
    subprocess.run(cmd, check=True)
    return beta, onsite_U, seed, output_path

results = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = []
    for u_idx, onsite_U in enumerate(onsite_Us):
        for beta_idx, beta in enumerate(betas):
            future = executor.submit(run_simulation, beta, onsite_U, beta_idx + 1, u_idx + 1)
            futures.append(future)
    for future in as_completed(futures):
        results.append(future.result())

print("Parallel execution complete.\n")

print("# inv_beta onsite_U x abs_err rel_err")
for beta, onsite_U, seed, path in results:
    out_files = glob.glob(os.path.join(path, f"{OBSERVABLE}_*.out"))
    if not out_files:
        print(f"# Warning: no files found for beta={beta:.6f} onsite_U={onsite_U:.6f}")
        continue
    for file in out_files:
        with open(file, "r") as f:
            line = f.readline().strip()
            try:
                x, abs_err, rel_err = map(float, line.split())
                print(f"{1/beta:.6f} {onsite_U:.6f} {x:.6f} {abs_err:.6f} {rel_err:.6f}")
            except ValueError:
                print(f"# Warning: could not parse {file}")

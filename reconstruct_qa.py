"""
Reconstructs empty QA .jsonl files from the individual .json result files
that ExperimentRunner saved to 03_results/raw/.
"""

import json
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("03_results/raw")

# Read all individual .json files (not .jsonl)
grouped = defaultdict(list)
for f in RAW_DIR.glob("*.json"):
    try:
        data = json.loads(f.read_text())
        config_name = data.get("config_name")
        if config_name:
            grouped[config_name].append(data)
    except Exception:
        continue

# Write grouped results into .jsonl files for QA configs
qa_configs = ["qa_zero_shot", "qa_few_shot_4shot", "qa_cot", "qa_zero_shot_cot"]

for config_name in qa_configs:
    rows = grouped.get(config_name, [])
    out_path = RAW_DIR / f"{config_name}.jsonl"
    if not rows:
        print(f"WARNING: No data found for {config_name}")
        continue
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Reconstructed {out_path.name} — {len(rows)} rows")

print("\nDone. Now run: python evaluate.py")

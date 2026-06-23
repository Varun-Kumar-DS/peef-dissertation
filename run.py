"""
PEEF — Main Experiment Runner
==============================
Run one experiment defined by a YAML config file.

Usage:
    python run.py --config 05_code/configs/qa_zero_shot.yaml
    python run.py --config 05_code/configs/pilot_qa_zero_shot.yaml
    python run.py --config 05_code/configs/reasoning_cot.yaml --n 20

Arguments:
    --config   Path to YAML config file (required)
    --n        Override sample size from config (optional)
    --dry-run  Build prompts but do NOT call the API (useful for testing)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

# Make sure 05_code is on the path so imports work
sys.path.insert(0, str(Path(__file__).parent / "05_code"))

from src.prompt_builder import PromptBuilder
from src.experiment_runner import ExperimentRunner
from utils.dataset_loader import load_dataset_for_task


# ---------------------------------------------------------------------------
# Few-shot example pools (small fixed sets, separate from the test set)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES: dict[str, list[dict]] = {
    "qa": [
        {"question": "What is the capital of France?",           "answer": "Paris",          "reasoning": "France is a country in Western Europe. Its capital city is Paris."},
        {"question": "Who wrote 'Romeo and Juliet'?",            "answer": "Shakespeare",     "reasoning": "Romeo and Juliet is a famous play written by William Shakespeare in the late 16th century."},
        {"question": "How many sides does a hexagon have?",      "answer": "Six",             "reasoning": "The prefix 'hex' means six, so a hexagon has six sides."},
        {"question": "What is the chemical symbol for gold?",    "answer": "Au",              "reasoning": "Gold's chemical symbol Au comes from the Latin word 'aurum'."},
        {"question": "Which planet is closest to the Sun?",      "answer": "Mercury",         "reasoning": "Mercury is the first planet from the Sun in our solar system."},
        {"question": "In what year did World War II end?",       "answer": "1945",            "reasoning": "World War II ended in 1945 with Germany surrendering in May and Japan in September."},
        {"question": "What is the largest ocean on Earth?",      "answer": "Pacific Ocean",   "reasoning": "The Pacific Ocean covers more than 30% of Earth's surface, making it the largest."},
        {"question": "Who painted the Mona Lisa?",               "answer": "Leonardo da Vinci","reasoning": "The Mona Lisa was painted by the Italian Renaissance artist Leonardo da Vinci."},
    ],
    "summarisation": [
        {
            "article": "Scientists at NASA have confirmed the discovery of water ice on the Moon's surface. The ice was found in permanently shadowed craters near the lunar poles. This discovery could be crucial for future human missions to the Moon, as the ice could be converted into drinking water or rocket fuel.",
            "summary": "NASA scientists have confirmed water ice exists in shadowed craters near the Moon's poles, a finding that could support future human lunar missions by providing water and rocket fuel.",
            "key_points": "1. Water ice confirmed on Moon. 2. Found in shadowed polar craters. 3. Could support future missions.",
        },
        {
            "article": "Electric vehicle sales surpassed 10 million units globally in 2023, representing 14% of all new car sales. China led the market with over 6 million EVs sold, followed by Europe and the United States. Battery costs have fallen 90% over the last decade, making EVs increasingly competitive with petrol cars.",
            "summary": "Global electric vehicle sales exceeded 10 million in 2023, with China dominating the market. Falling battery costs, down 90% over a decade, are making EVs increasingly cost-competitive with traditional vehicles.",
            "key_points": "1. 10 million EVs sold globally in 2023. 2. China leads with 6 million units. 3. Battery costs fell 90% over 10 years.",
        },
    ],
    "reasoning": [
        {"problem": "If a train travels at 60 mph for 2 hours, how far does it travel?",      "answer": "120", "working": "Distance = speed × time = 60 × 2 = 120 miles."},
        {"problem": "A store sells apples for £0.50 each. How much do 8 apples cost?",        "answer": "4",   "working": "Cost = 0.50 × 8 = £4.00."},
        {"problem": "If there are 24 students in a class and 1/4 are absent, how many are present?", "answer": "18", "working": "Absent = 24 × 1/4 = 6. Present = 24 - 6 = 18."},
        {"problem": "A rectangle has a length of 8 cm and a width of 5 cm. What is its area?","answer": "40",  "working": "Area = length × width = 8 × 5 = 40 cm²."},
        {"problem": "Tom has 15 sweets and gives 3 to each of his 4 friends. How many does he have left?", "answer": "3", "working": "Given away = 3 × 4 = 12. Remaining = 15 - 12 = 3."},
        {"problem": "If a pizza has 8 slices and 3 people each eat 2 slices, how many slices remain?", "answer": "2", "working": "Eaten = 3 × 2 = 6. Remaining = 8 - 6 = 2."},
        {"problem": "A car uses 5 litres of fuel per 100 km. How much fuel is needed for 300 km?", "answer": "15", "working": "Fuel = (5 / 100) × 300 = 15 litres."},
        {"problem": "If 6 workers can build a wall in 10 days, how many days would 3 workers take?", "answer": "20", "working": "Total work = 6 × 10 = 60 worker-days. Days for 3 workers = 60 / 3 = 20."},
    ],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PEEF experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--n",      type=int, default=None, help="Override sample size")
    parser.add_argument("--dry-run", action="store_true", help="Build prompts only, no API calls")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override sample size if --n is given
    if args.n is not None:
        config["sample_size"] = args.n

    task      = config["task"]
    technique = config["technique"]
    n_shots   = config.get("n_shots", 0)
    n_samples = config.get("sample_size", 50)
    split     = config.get("dataset_split", "validation")

    print(f"\n{'='*60}")
    print(f"  PEEF Experiment: {config.get('name', config_path.stem)}")
    print(f"  Task: {task} | Technique: {technique} | Shots: {n_shots}")
    print(f"  Model: {config.get('model', 'claude-haiku-4-5-20251001')} | Samples: {n_samples}")
    print(f"{'='*60}\n")

    # 1. Load dataset
    print(f"Loading dataset ({task}, {split}, n={n_samples})...")
    items = load_dataset_for_task(task, split=split, n=n_samples)
    print(f"  Loaded {len(items)} items.\n")

    # 2. Build prompts
    examples = FEW_SHOT_EXAMPLES.get(task, [])
    builder = PromptBuilder(task=task, technique=technique, n_shots=n_shots, examples=examples)

    input_field = {"qa": "question", "summarisation": "article", "reasoning": "problem"}[task]

    prompt_items = []
    for item in items:
        prompt = builder.build(**{input_field: item[input_field]})
        run_id = hashlib.md5(f"{config.get('name', '')}::{item['id']}".encode()).hexdigest()[:12]
        prompt_items.append({
            "run_id": run_id,
            "prompt": prompt,
            "reference": item["reference"],
            "item_id": item["id"],
            "metadata": {"item_id": item["id"], "reference": item["reference"]},
        })

    if args.dry_run:
        print("DRY RUN — first 2 prompts:\n")
        for p in prompt_items[:2]:
            print(f"--- {p['item_id']} ---")
            print(p["prompt"])
            print(f"\nReference: {p['reference']}\n")
        print(f"Total prompts built: {len(prompt_items)}")
        return

    # 3. Run experiment
    output_dir = Path(config.get("output_file", f"03_results/raw/{config_path.stem}.jsonl")).parent
    runner = ExperimentRunner(config=config, output_dir=output_dir)
    print(f"Running {len(prompt_items)} prompts against the API...\n")
    results = runner.run(prompt_items)

    # 4. Save summary
    output_file = Path(config.get("output_file", f"03_results/raw/{config_path.stem}.jsonl"))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for r in results:
            import dataclasses
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")

    total_tokens = sum(r.total_tokens for r in results)
    cost_usd     = total_tokens / 1_000_000 * 0.80   # Haiku input price (conservative)
    print(f"\n{'='*60}")
    print(f"  Done! {len(results)} results saved to {output_file}")
    print(f"  Total tokens used : {total_tokens:,}")
    print(f"  Estimated cost    : ${cost_usd:.4f} (~£{cost_usd * 0.79:.4f})")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

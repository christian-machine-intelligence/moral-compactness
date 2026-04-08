"""
Phase 3: Statistical analysis of scored trials.

Computes:
- Scheming rates per cell (model x condition)
- Reasoned-past rates per cell
- Two-proportion z-tests between conditions
- Cohen's h effect sizes
- Summary tables
"""

import argparse
import json
import math
from pathlib import Path

from scipy.stats import norm
from tabulate import tabulate

ROOT = Path(__file__).parent.parent
SCORED_DIR = ROOT / "results" / "scored"


def two_proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> float:
    """Two-proportion z-test, returns two-sided p-value."""
    if n1 == 0 or n2 == 0:
        return 1.0
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    if p_pool == 0 or p_pool == 1:
        return 1.0
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return 2 * (1 - norm.cdf(abs(z)))


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def load_scored_trials() -> list[dict]:
    """Load all scored trial files."""
    trials = []
    for f in sorted(SCORED_DIR.rglob("trial_*.json")):
        with open(f) as fh:
            trial = json.load(fh)
            if trial.get("score") is not None:
                trials.append(trial)
    return trials


def compute_rates(trials: list[dict]) -> dict:
    """Compute scheming and reasoned-past rates grouped by model x condition."""
    cells = {}
    for t in trials:
        key = (t["model"], t["condition"])
        if key not in cells:
            cells[key] = {"scheming": [], "reasoned_past": []}

        score = t["score"]
        cells[key]["scheming"].append(score["scheming"])
        if score["reasoned_past"] is not None:
            cells[key]["reasoned_past"].append(score["reasoned_past"])

    rates = {}
    for key, data in cells.items():
        n_scheme = len(data["scheming"])
        n_rp = len(data["reasoned_past"])
        rates[key] = {
            "scheming_rate": sum(data["scheming"]) / n_scheme if n_scheme else 0,
            "scheming_n": n_scheme,
            "reasoned_past_rate": sum(data["reasoned_past"]) / n_rp if n_rp else 0,
            "reasoned_past_n": n_rp,
        }
    return rates


def print_scheming_table(rates: dict) -> None:
    """Print scheming rate comparison table."""
    print("\n" + "=" * 70)
    print("SCHEMING RATES")
    print("=" * 70)

    rows = []
    for (model, condition), r in sorted(rates.items()):
        rows.append([
            model,
            condition,
            f"{r['scheming_rate']:.1%}",
            r["scheming_n"],
        ])

    print(tabulate(rows, headers=["Model", "Condition", "Scheming Rate", "N"],
                   tablefmt="grid"))


def print_reasoned_past_table(rates: dict) -> None:
    """Print reasoned-past rate comparison table."""
    print("\n" + "=" * 70)
    print("REASONED-PAST RATES (among trials with visible thinking)")
    print("=" * 70)

    rows = []
    for (model, condition), r in sorted(rates.items()):
        if r["reasoned_past_n"] > 0:
            rows.append([
                model,
                condition,
                f"{r['reasoned_past_rate']:.1%}",
                r["reasoned_past_n"],
            ])

    print(tabulate(rows, headers=["Model", "Condition", "Reasoned Past", "N"],
                   tablefmt="grid"))


def print_pairwise_tests(rates: dict) -> None:
    """Print pairwise comparisons between conditions, per model."""
    print("\n" + "=" * 70)
    print("PAIRWISE COMPARISONS: Elaborate Rules vs. Deontological Absolutes")
    print("=" * 70)

    models = sorted(set(m for m, _ in rates.keys()))

    for model in models:
        rules = rates.get((model, "elaborate_rules"))
        deont = rates.get((model, "deontological_absolutes"))
        baseline = rates.get((model, "minimal_baseline"))

        if not rules or not deont:
            continue

        print(f"\n--- {model} ---")
        rows = []

        # Scheming rate comparison
        p_val = two_proportion_z_test(
            rules["scheming_rate"], rules["scheming_n"],
            deont["scheming_rate"], deont["scheming_n"],
        )
        h = cohens_h(rules["scheming_rate"], deont["scheming_rate"])
        rows.append([
            "Scheming rate",
            f"{rules['scheming_rate']:.1%}",
            f"{deont['scheming_rate']:.1%}",
            f"{baseline['scheming_rate']:.1%}" if baseline else "—",
            f"{p_val:.4f}",
            f"{h:+.3f}",
            "*" if p_val < 0.05 else "",
        ])

        # Reasoned-past comparison
        if rules["reasoned_past_n"] > 0 and deont["reasoned_past_n"] > 0:
            p_val_rp = two_proportion_z_test(
                rules["reasoned_past_rate"], rules["reasoned_past_n"],
                deont["reasoned_past_rate"], deont["reasoned_past_n"],
            )
            h_rp = cohens_h(rules["reasoned_past_rate"], deont["reasoned_past_rate"])
            rows.append([
                "Reasoned-past rate",
                f"{rules['reasoned_past_rate']:.1%}",
                f"{deont['reasoned_past_rate']:.1%}",
                f"{baseline['reasoned_past_rate']:.1%}" if baseline and baseline['reasoned_past_n'] > 0 else "—",
                f"{p_val_rp:.4f}",
                f"{h_rp:+.3f}",
                "*" if p_val_rp < 0.05 else "",
            ])

        print(tabulate(
            rows,
            headers=["Metric", "Rules", "Deontological", "Baseline", "p-value", "Cohen's h", "Sig"],
            tablefmt="grid",
        ))


def print_example_reasoning(trials: list[dict], n: int = 3) -> None:
    """Print example reasoning excerpts for qualitative inspection."""
    print("\n" + "=" * 70)
    print("EXAMPLE REASONING EXCERPTS")
    print("=" * 70)

    for condition in ["elaborate_rules", "deontological_absolutes", "minimal_baseline"]:
        cond_trials = [t for t in trials if t["condition"] == condition and t.get("score")]
        scheming_trials = [t for t in cond_trials if t["score"]["scheming"]]
        non_scheming = [t for t in cond_trials if not t["score"]["scheming"]]

        print(f"\n--- {condition} ---")

        if scheming_trials:
            t = scheming_trials[0]
            print(f"\n  [SCHEMED] (model: {t['model']}, trial: {t['trial']})")
            if t.get("thinking"):
                excerpt = t["thinking"][:500]
                print(f"  Thinking: {excerpt}...")
            print(f"  Score: {json.dumps(t['score'], indent=4)}")

        if non_scheming:
            t = non_scheming[0]
            print(f"\n  [DID NOT SCHEME] (model: {t['model']}, trial: {t['trial']})")
            if t.get("thinking"):
                excerpt = t["thinking"][:500]
                print(f"  Thinking: {excerpt}...")
            print(f"  Score: {json.dumps(t['score'], indent=4)}")


def analyze():
    trials = load_scored_trials()
    if not trials:
        print("No scored trials found in results/scored/")
        return

    print(f"Loaded {len(trials)} scored trials.")
    rates = compute_rates(trials)

    print_scheming_table(rates)
    print_reasoned_past_table(rates)
    print_pairwise_tests(rates)
    print_example_reasoning(trials)

    # Save summary JSON
    summary_path = ROOT / "results" / "analysis_summary.json"
    summary = {}
    for (model, condition), r in rates.items():
        summary[f"{model}/{condition}"] = r
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


def main():
    analyze()


if __name__ == "__main__":
    main()

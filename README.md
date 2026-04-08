![Saint Jerome in His Study, Antonello da Messina (1475)](banner.jpg)

# Moral Compactness

**ICMI Working Paper No. 10**

Code, data, and analysis for "Moral Compactness: Scripture as a Kolmogorov-Efficient Constraint for LLM Scheming."

**Read the paper:** [ICMI Proceedings Working Paper No. 10](https://icmi-proceedings.com/ICMI-010-moral-compactness.html)

## Quick Start

```bash
cp .env.example .env
# Add your API keys to .env

pip install -r requirements.txt

# Generate trials
python3 -m src generate --model gpt-5.4 --condition elaborate_rules --n 100

# Score with judge
python3 -m src score

# Run analysis
python3 -m src analyze
```

## Structure

```
configs/
  constraints/         # All constraint prompt conditions
  scenarios/           # Scheming evaluation scenario (YAML)
src/
  generate_data.py     # Trial generation (async, with resume)
  score.py             # LLM judge scoring
  analysis.py          # Statistical analysis
  runners/             # API runners (Anthropic, OpenAI)
results/
  raw/                 # Raw trial outputs (JSON per trial)
  scored/              # Judge-scored trials
```

## Conditions

| Condition | Description | Words |
|-----------|-------------|------:|
| `elaborate_rules` | 24 enumerated secular rules | ~2,000 |
| `deontological_religious` | Scripture-based moral framework | ~250 |
| `elaborate_rules_ablated` | Rules without affirmative disclosure duty | ~1,800 |
| `deontological_religious_ablated` | Religious framework without James 4:17 | ~200 |
| `minimal_baseline` | No constraints | ~10 |

## Citation

```bibtex
@article{hwang2026moralcompactness,
  title={Moral Compactness: Scripture as a Kolmogorov-Efficient Constraint for LLM Scheming},
  author={Hwang, Tim},
  journal={ICMI Working Paper No. 10},
  year={2026}
}
```

"""Bootstrap intervals for the model comparisons reported in Section 3.1.1.

Run from the repository root:  python src/model_comparison_bootstrap.py
Resamples the 1,273 tweets with replacement (the same tweets for every model) and reports
(i) 95% intervals for macro-F1 for every model x prompt, (ii) the paired Gemini-minus-Flan-T5
macro-F1 difference per prompt, (iii) few-shot-minus-roleplay for Gemini and Flan-T5.
Gemini uses the first of its three samples, as in the voting analysis.
"""
import ast
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

PROMPTS = ["roleplay", "content", "zero-shot", "few-shot"]
FILES = {"ChatGPT": "gpt", "Flan-T5": "flan_t5", "Alpaca": "alpaca", "Vicuna": "vicuna", "Mistral": "mistral"}
GEMINI_COLS = {"roleplay": "roleplay_results", "content": "content_results",
               "zero-shot": "zeroshot_results", "few-shot": "fewshot_results"}
N_BOOT, SEED = 1000, 7


def to_binary(s):
    s = str(s).lower().replace("'", "").replace('"', "").strip()
    return 0 if s.startswith("not") else (1 if "sexist" in s else 0)


preds = {}
for name, f in FILES.items():
    d = pd.read_csv(f"data/llm_data/{f}.csv")
    y = d["label"].values
    for p in PROMPTS:
        preds[(name, p)] = d[p].apply(to_binary).values
g = pd.read_csv("data/llm_data/gemini.csv")
assert (g["uniqueID"].values == d["uniqueID"].values).all()
for p in PROMPTS:
    preds[("Gemini", p)] = g[GEMINI_COLS[p]].apply(lambda s: to_binary(ast.literal_eval(s)[0])).values

rng = np.random.default_rng(SEED)
n = len(y)
idx = [rng.integers(0, n, n) for _ in range(N_BOOT)]
f1 = lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0)
boot = {k: np.array([f1(y[i], v[i]) for i in idx]) for k, v in preds.items()}

print("Macro-F1 (%) with 95% bootstrap interval")
for k, v in preds.items():
    lo, hi = np.percentile(boot[k], [2.5, 97.5])
    print(k, round(100 * f1(y, v), 1), f"[{100*lo:.1f}, {100*hi:.1f}]")
print("\nGemini minus Flan-T5 (paired)")
for p in PROMPTS:
    d_ = boot[("Gemini", p)] - boot[("Flan-T5", p)]
    lo, hi = np.percentile(d_, [2.5, 97.5])
    point = f1(y, preds[("Gemini", p)]) - f1(y, preds[("Flan-T5", p)])
    print(p, round(100 * point, 1), f"[{100*lo:.1f}, {100*hi:.1f}]")
for m in ["Gemini", "Flan-T5"]:
    d_ = boot[(m, "few-shot")] - boot[(m, "roleplay")]
    lo, hi = np.percentile(d_, [2.5, 97.5])
    point = f1(y, preds[(m, "few-shot")]) - f1(y, preds[(m, "roleplay")])
    print(f"{m} few-shot minus roleplay", round(100 * point, 1), f"[{100*lo:.1f}, {100*hi:.1f}]")
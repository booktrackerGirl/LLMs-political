# ============================================================
# FM4: SEED STABILITY, SEQUENCE PROBABILITY, AND CORRECTNESS
# ============================================================
#
# Dataset structure:
#   3 CSV files = 3 different random seeds
#   Each CSV contains the same 1,273 tweets
#   Each CSV contains 4 prompt conditions:
#       roleplay
#       content
#       zero-shot
#       few-shot
#
# Main outputs:
#   1. Data integrity checks
#   2. Per-seed correctness
#   3. Probability-vs-correctness correlations
#   4. Tweet-level bootstrap CIs
#   5. Logistic regression with clustered SEs
#   6. AUROC
#   7. Spearman correlation
#   8. Probability-bin diagnostics
#   9. Per-tweet 3-seed JSD
#  10. JSD summary statistics
#  11. Seed agreement
#  12. Stable-but-wrong analysis
#  13. Figures
#  14. Paper-ready CSV tables
#
# IMPORTANT:
# The three seeds are repeated measurements of the SAME tweets.
# We therefore do NOT treat 3 x 1,273 observations as 3,819
# independent tweets for inferential statistics.
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================

import os
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import (
    pointbiserialr,
    spearmanr,
    pearsonr,
    mannwhitneyu,
    wilcoxon
)

from scipy.special import expit
from scipy.optimize import brentq

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)

import statsmodels.api as sm
import statsmodels.formula.api as smf


warnings.filterwarnings("ignore")


# ============================================================
# 1. USER SETTINGS
# ============================================================

# ------------------------------------------------------------
# CHANGE THESE THREE FILENAMES IF NECESSARY
# ------------------------------------------------------------

FILES = [
    "data/entropy_data/flan1.csv",
    "data/entropy_data/flan2.csv",
    "data/entropy_data/flan3.csv",
    "data/entropy_data/flan4.csv"
]

# Output directory
OUTPUT_DIR = Path("FM4_outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)




# ------------------------------------------------------------
# PROMPT NAMES
# ------------------------------------------------------------

PROMPTS = [
    "roleplay",
    "content",
    "zero-shot",
    "few-shot"
]


for path in FILES:
    df = pd.read_csv(path)
    print(
        path,
        df.loc[823, ["uniqueID", "trigger_event", "label"]].to_dict()
    )


# ------------------------------------------------------------
# LABEL CODING
# ------------------------------------------------------------

LABEL_MAP = {
    "not sexist": 0,
    "sexist": 1
}


# ------------------------------------------------------------
# PROBABILITY BIN EDGES
#
# These are descriptive bins, NOT a formal calibration curve.
# ------------------------------------------------------------

PROB_BINS = [
    0.00,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
    1.0000001
]


# ------------------------------------------------------------
# BOOTSTRAP SETTINGS
# ------------------------------------------------------------

N_BOOT = 5000
RANDOM_SEED = 20270915

rng = np.random.default_rng(RANDOM_SEED)


# ============================================================
# 2. BASIC FUNCTIONS
# ============================================================

def parse_prediction_cell(x):
    """
    Parse prediction cells of the form:
        ('not sexist', 0.9166759)
        ('sexist', 0.9298217)

    If only the predicted label is present, e.g.:
        'not sexist'
    return the label and NaN for probability.

    Missing probabilities are retained for prediction analyses but
    excluded from probability-based FM4 analyses.
    """
    import ast
    import numpy as np

    if pd.isna(x):
        return None, np.nan

    text = str(x).strip()

    # Normal expected format:
    # ('not sexist', 0.9166759)
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (tuple, list)) and len(parsed) >= 2:
            pred = str(parsed[0]).strip().lower()
            prob = float(parsed[1])
            return pred, prob

    except (ValueError, SyntaxError, TypeError):
        pass

    # Missing-probability format:
    # 'not sexist'
    # 'sexist'
    normalized = text.lower()

    if normalized in {"sexist", "not sexist"}:
        return normalized, np.nan

    raise ValueError(
        f"Could not parse prediction cell:\n"
        f"{text}"
    )




def prediction_to_binary(pred):
    """
    Convert:
        sexist -> 1
        not sexist -> 0
    """

    if pd.isna(pred):
        return np.nan

    pred = str(pred).strip().lower()

    if pred not in LABEL_MAP:
        raise ValueError(f"Unknown prediction: {pred}")

    return LABEL_MAP[pred]


def probability_of_sexist(pred, prob):
    """
    Convert probability assigned to the predicted label
    into probability assigned specifically to 'sexist'.

    Example:

        prediction = sexist
        prob = 0.92

        P(sexist) = 0.92

    whereas:

        prediction = not sexist
        prob = 0.92

        P(sexist) = 1 - 0.92 = 0.08
    """

    if pd.isna(pred) or pd.isna(prob):
        return np.nan

    if pred == "sexist":
        return prob

    elif pred == "not sexist":
        return 1.0 - prob

    else:
        raise ValueError(pred)


def probability_of_predicted_class(pred, prob):
    """
    Simply returns the supplied probability.

    This is the quantity corresponding to the model's
    probability for the class it selected.
    """

    return prob


def make_distribution(pred, prob):
    """
    Convert prediction + probability into a binary class
    probability distribution:

        [P(not sexist), P(sexist)]

    Example:

        ('sexist', 0.90)
        -> [0.10, 0.90]

        ('not sexist', 0.90)
        -> [0.90, 0.10]
    """

    if pred == "sexist":
        return np.array([1.0 - prob, prob])

    elif pred == "not sexist":
        return np.array([prob, 1.0 - prob])

    else:
        raise ValueError(pred)


# ============================================================
# 3. LOAD THE FOUR SEED FILES
# ============================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

raw_runs = {}

for seed_number, filename in enumerate(FILES, start=1):

    print(f"\nLoading seed {seed_number}: {filename}")

    df = pd.read_csv(filename)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    required_columns = [
        "uniqueID",
        "trigger_event",
        "label"
    ] + PROMPTS

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Seed {seed_number} is missing columns: {missing}"
        )

    raw_runs[f"seed_{seed_number}"] = df.copy()




# ============================================================
# 4. CHECK DATA STRUCTURE
# ============================================================

print("\n" + "=" * 80)
print("DATA INTEGRITY CHECKS")
print("=" * 80)


# ------------------------------------------------------------
# Number of observations
# ------------------------------------------------------------

for seed_name, df in raw_runs.items():

    print(
        f"{seed_name}: "
        f"{len(df):,} rows | "
        f"{df['uniqueID'].nunique():,} unique IDs"
    )

    if df["uniqueID"].duplicated().any():
        duplicates = df.loc[
            df["uniqueID"].duplicated(),
            "uniqueID"
        ].tolist()

        raise ValueError(
            f"{seed_name} contains duplicate uniqueID values: "
            f"{duplicates[:10]}"
        )


# ------------------------------------------------------------
# Check that all seeds contain exactly the same tweets
# ------------------------------------------------------------

id_sets = {
    seed_name: set(df["uniqueID"])
    for seed_name, df in raw_runs.items()
}

reference_ids = id_sets["seed_1"]

for seed_name, ids in id_sets.items():

    missing = reference_ids - ids
    extra = ids - reference_ids

    print(f"\n{seed_name}")

    print(f"  Missing IDs: {len(missing)}")
    print(f"  Extra IDs:   {len(extra)}")

    if missing:
        print("  Example missing:", list(missing)[:10])

    if extra:
        print("  Example extra:", list(extra)[:10])


# ------------------------------------------------------------
# Check labels are identical across seeds
# ------------------------------------------------------------

print("\nChecking labels across seeds...")

label_reference = (
    raw_runs["seed_1"]
    .set_index("uniqueID")["label"]
)

for seed_name, df in raw_runs.items():

    labels = (
        df.set_index("uniqueID")["label"]
        .reindex(label_reference.index)
    )

    mismatches = labels != label_reference

    print(
        f"{seed_name}: "
        f"{mismatches.sum()} label mismatches"
    )

    if mismatches.any():
        raise ValueError(
            f"Labels differ across seeds for {seed_name}."
        )


# ============================================================
# 5. NORMALISE LABELS
# ============================================================

def normalise_gold_label(x):

    x = str(x).strip().lower()

    if x in ["sexist", "1", "true"]:
        return 1

    elif x in [
        "not sexist",
        "not_sexist",
        "non-sexist",
        "non sexist",
        "0",
        "false"
    ]:
        return 0

    else:
        raise ValueError(
            f"Unknown gold label: {x}"
        )


gold_df = raw_runs["seed_1"][
    ["uniqueID", "trigger_event", "label"]
].copy()

gold_df["gold"] = gold_df["label"].apply(
    normalise_gold_label
)

gold_df = gold_df.drop(columns=["label"])

# ============================================================
# CHECK WHETHER SEED RUNS ARE ACTUALLY DIFFERENT
# ============================================================

print("\n" + "=" * 80)
print("SEED DIFFERENCE CHECK")
print("=" * 80)

for i in range(len(raw_runs)):
    for j in range(i + 1, len(raw_runs)):

        seed_names = list(raw_runs.keys())
        seed_a = seed_names[i]
        seed_b = seed_names[j]

        df_a = raw_runs[seed_a]
        df_b = raw_runs[seed_b]

        print(f"\n{seed_a} vs {seed_b}")

        for prompt in PROMPTS:

            same = (
                df_a[prompt].astype(str).reset_index(drop=True)
                ==
                df_b[prompt].astype(str).reset_index(drop=True)
            )

            print(
                f"  {prompt}: "
                f"{same.sum():,}/{len(same):,} identical "
                f"({same.mean()*100:.2f}%)"
            )



# ============================================================
# 6. PARSE EVERY SEED × PROMPT
# ============================================================

long_rows = []

for seed_name, df in raw_runs.items():

    for _, row in df.iterrows():

        for prompt in PROMPTS:

            pred, prob = parse_prediction_cell(
                row[prompt]
            )

            gold = normalise_gold_label(row["label"])

            pred_binary = prediction_to_binary(pred)

            correct = int(
                pred_binary == gold
            )

            p_sexist = probability_of_sexist(
                pred,
                prob
            )

            long_rows.append({

                "seed": seed_name,

                "uniqueID": row["uniqueID"],

                "trigger_event": row["trigger_event"],

                "gold": gold,

                "prompt": prompt,

                "prediction": pred,

                "prediction_binary": pred_binary,

                "predicted_class_probability": prob,

                "p_sexist": p_sexist,

                "correct": correct
            })



long_df = pd.DataFrame(long_rows)

print("\nLONG_DF COLUMNS:")
print(long_df.columns.tolist())

missing_prob = long_df["predicted_class_probability"].isna()

print("\n" + "=" * 80)
print("MISSING PROBABILITY CHECK")
print("=" * 80)

print("Rows with missing probability:", missing_prob.sum())

if missing_prob.any():
    print(
        long_df.loc[
            missing_prob,
            ["seed", "uniqueID", "prompt", "prediction"]
        ].to_string(index=False)
    )


print(
    f"\nLong-format dataset: "
    f"{len(long_df):,} rows"
)

print(
    f"Expected: "
    f"{3 * 1273 * 4:,} rows"
)


# ============================================================
# 7. SAVE CLEAN MASTER DATA
# ============================================================

long_df.to_csv(
    OUTPUT_DIR / "FM4_master_long.csv",
    index=False
)


# ============================================================
# 8. BASIC PERFORMANCE BY SEED AND PROMPT
# ============================================================

performance_rows = []

for seed in long_df["seed"].unique():

    for prompt in PROMPTS:

        sub = long_df[
            (long_df["seed"] == seed) &
            (long_df["prompt"] == prompt)
        ]

        performance_rows.append({

            "seed": seed,

            "prompt": prompt,

            "N": len(sub),

            "accuracy": sub["correct"].mean(),

            "mean_predicted_probability":
                sub["predicted_class_probability"].mean(),

            "median_predicted_probability":
                sub["predicted_class_probability"].median(),

            "sexist_predictions":
                (sub["prediction"] == "sexist").sum(),

            "sexist_prediction_rate":
                (sub["prediction"] == "sexist").mean(),

            "gold_sexist_rate":
                sub["gold"].mean()
        })


performance_df = pd.DataFrame(performance_rows)

performance_df.to_csv(
    OUTPUT_DIR / "FM4_performance_by_seed_prompt.csv",
    index=False
)

print("\nPerformance by seed and prompt:")
print(
    performance_df.to_string(index=False)
)


# ============================================================
# 9. POINT-BISERIAL CORRELATION
# ============================================================
#
# IMPORTANT:
#
# This is calculated separately for every seed.
#
# We do NOT pool the 3 seeds as independent observations.
#
# The variable:
#
#     predicted_class_probability
#
# is the probability assigned to the model's predicted class.
#
# The variable:
#
#     correct
#
# is 1 for a correct prediction and 0 otherwise.
#
# A negative correlation means:
#
#     higher model-assigned probability
#     -> lower likelihood of correctness
#
# ============================================================

correlation_rows = []

for seed in long_df["seed"].unique():

    for prompt in PROMPTS:

        sub = long_df[
            (long_df["seed"] == seed) &
            (long_df["prompt"] == prompt)
        ].copy()

        # Exclude observations with missing probabilities
        corr_data = sub[
            sub["predicted_class_probability"].notna()
            & sub["correct"].notna()
        ].copy()

        r, p = pointbiserialr(
            corr_data["correct"],
            corr_data["predicted_class_probability"]
        )

        rho, rho_p = spearmanr(
            sub["correct"],
            sub["predicted_class_probability"]
        )

        correlation_rows.append({

            "seed": seed,

            "prompt": prompt,

            "N": len(sub),

            "point_biserial_r": r,

            "point_biserial_p": p,

            "spearman_rho": rho,

            "spearman_p": rho_p
        })


correlation_df = pd.DataFrame(
    correlation_rows
)

correlation_df.to_csv(
    OUTPUT_DIR / "FM4_probability_correctness_correlations.csv",
    index=False
)

print("\nProbability/correctness correlations:")
print(
    correlation_df.to_string(index=False)
)


# ============================================================
# 10. SUMMARY ACROSS SEEDS
# ============================================================

correlation_summary = (
    correlation_df
    .groupby("prompt")
    .agg(
        mean_r=("point_biserial_r", "mean"),
        median_r=("point_biserial_r", "median"),
        min_r=("point_biserial_r", "min"),
        max_r=("point_biserial_r", "max"),

        mean_spearman=("spearman_rho", "mean"),
        min_spearman=("spearman_rho", "min"),
        max_spearman=("spearman_rho", "max")
    )
    .reset_index()
)

correlation_summary.to_csv(
    OUTPUT_DIR / "FM4_correlation_summary_across_seeds.csv",
    index=False
)


# ============================================================
# 11. TWEET-LEVEL BOOTSTRAP
# ============================================================
#
# For each seed × prompt:
#
#   sample tweets WITH replacement
#
# This preserves the correct unit of analysis.
#
# ============================================================

def bootstrap_point_biserial(
    df,
    n_boot=5000,
    seed=20270915
):

    local_rng = np.random.default_rng(seed)

    n = len(df)

    x = df["predicted_class_probability"].to_numpy()
    y = df["correct"].to_numpy()

    observed, _ = pointbiserialr(y, x)

    boot = []

    for _ in range(n_boot):

        idx = local_rng.integers(
            0,
            n,
            size=n
        )

        xb = x[idx]
        yb = y[idx]

        # Need both classes present
        if len(np.unique(yb)) < 2:
            continue

        r, _ = pointbiserialr(yb, xb)

        if np.isfinite(r):
            boot.append(r)

    boot = np.asarray(boot)

    lower = np.percentile(
        boot,
        2.5
    )

    upper = np.percentile(
        boot,
        97.5
    )

    return observed, lower, upper, boot


bootstrap_rows = []

for seed in long_df["seed"].unique():

    for prompt in PROMPTS:

        sub = long_df[
            (long_df["seed"] == seed) &
            (long_df["prompt"] == prompt)
        ].copy()

        r, lower, upper, boot = (
            bootstrap_point_biserial(
                sub,
                n_boot=N_BOOT,
                seed=RANDOM_SEED
            )
        )

        bootstrap_rows.append({

            "seed": seed,

            "prompt": prompt,

            "N": len(sub),

            "r": r,

            "bootstrap_CI_lower": lower,

            "bootstrap_CI_upper": upper,

            "bootstrap_N": len(boot)
        })


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)

bootstrap_df.to_csv(
    OUTPUT_DIR / "FM4_bootstrap_pointbiserial_CI.csv",
    index=False
)


# ============================================================
# 12. LOGISTIC REGRESSION
# ============================================================
#
# Outcome:
#
#     correct
#
# Predictor:
#
#     predicted_class_probability
#
# Since each tweet appears once per seed, observations are
# clustered by uniqueID.
#
# We therefore use cluster-robust standard errors.
#
# This model asks:
#
#     Is higher assigned probability associated with the
#     probability of being correct?
#
# ============================================================

logistic_rows = []

for prompt in PROMPTS:

    sub = long_df[
        long_df["prompt"] == prompt
    ].copy()

    X = sm.add_constant(
        sub["predicted_class_probability"]
    )

    y = sub["correct"]

    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial()
    )

    result = model.fit(
        cov_type="cluster",
        cov_kwds={
            "groups": sub["uniqueID"]
        }
    )

    beta = result.params[
        "predicted_class_probability"
    ]

    se = result.bse[
        "predicted_class_probability"
    ]

    p = result.pvalues[
        "predicted_class_probability"
    ]

    ci_low, ci_high = (
        result.conf_int()
        .loc["predicted_class_probability"]
    )

    logistic_rows.append({

        "prompt": prompt,

        "N": len(sub),

        "beta": beta,

        "SE": se,

        "p": p,

        "CI_lower": ci_low,

        "CI_upper": ci_high,

        "odds_ratio_per_1_unit":
            np.exp(beta),

        "OR_CI_lower":
            np.exp(ci_low),

        "OR_CI_upper":
            np.exp(ci_high)
    })


logistic_df = pd.DataFrame(
    logistic_rows
)

logistic_df.to_csv(
    OUTPUT_DIR / "FM4_logistic_clustered.csv",
    index=False
)

print("\nClustered logistic regression:")
print(
    logistic_df.to_string(index=False)
)


# ============================================================
# 13. AUROC
# ============================================================
#
# Here:
#
#   score = predicted-class probability
#   target = correctness
#
# AUROC = 0.5
#     no ranking relationship
#
# AUROC > 0.5
#     higher probability tends to identify correct predictions
#
# AUROC < 0.5
#     higher probability tends to identify incorrect predictions
#
# IMPORTANT:
# This is NOT ordinary classification AUROC.
# It evaluates probability as a signal of CORRECTNESS.
#
# ============================================================

auc_rows = []

for seed in long_df["seed"].unique():

    for prompt in PROMPTS:

        sub = long_df[
            (long_df["seed"] == seed) &
            (long_df["prompt"] == prompt)
        ]

        auc = roc_auc_score(
            sub["correct"],
            sub["predicted_class_probability"]
        )

        ap = average_precision_score(
            sub["correct"],
            sub["predicted_class_probability"]
        )

        auc_rows.append({

            "seed": seed,

            "prompt": prompt,

            "N": len(sub),

            "AUROC_correctness": auc,

            "AveragePrecision_correctness": ap
        })


auc_df = pd.DataFrame(auc_rows)

auc_df.to_csv(
    OUTPUT_DIR / "FM4_AUROC_correctness.csv",
    index=False
)


# ============================================================
# 14. PROBABILITY-BIN DIAGNOSTICS
# ============================================================
#
# These are deliberately descriptive.
#
# DO NOT call this Expected Calibration Error unless you
# redesign the analysis specifically as a calibration study.
#
# The quantity below is:
#
#     empirical correctness rate
#     among predictions in a probability bin.
#
# ============================================================

bin_rows = []

for seed in long_df["seed"].unique():

    for prompt in PROMPTS:

        sub = long_df[
            (long_df["seed"] == seed) &
            (long_df["prompt"] == prompt)
        ].copy()

        sub["prob_bin"] = pd.cut(
            sub["predicted_class_probability"],
            bins=PROB_BINS,
            right=False,
            include_lowest=True
        )

        grouped = (
            sub
            .groupby(
                "prob_bin",
                observed=False
            )
            .agg(
                N=("correct", "size"),
                correct=("correct", "sum"),
                empirical_correctness=(
                    "correct",
                    "mean"
                ),
                mean_probability=(
                    "predicted_class_probability",
                    "mean"
                )
            )
            .reset_index()
        )

        grouped["seed"] = seed
        grouped["prompt"] = prompt

        bin_rows.append(grouped)


bins_df = pd.concat(
    bin_rows,
    ignore_index=True
)

bins_df.to_csv(
    OUTPUT_DIR / "FM4_probability_bins.csv",
    index=False
)


# ============================================================
# 15. PREPARE THREE-SEED DATA FOR JSD
# ============================================================
#
# For every:
#
#     tweet × prompt
#
# we want:
#
#     Seed 1 distribution
#     Seed 2 distribution
#     Seed 3 distribution
#
# Each distribution is:
#
#     [P(not sexist), P(sexist)]
#
# ============================================================

wide_rows = []

for prompt in PROMPTS:

    temp = long_df[
        long_df["prompt"] == prompt
    ].copy()

    pivot = temp.pivot(
        index="uniqueID",
        columns="seed",
        values=[
            "prediction",
            "predicted_class_probability",
            "correct",
            "p_sexist"
        ]
    )

    pivot.columns = [
        f"{a}_{b}"
        for a, b in pivot.columns
    ]

    pivot = pivot.reset_index()

    # Add gold labels
    pivot = pivot.merge(
        gold_df[
            ["uniqueID", "trigger_event", "gold"]
        ],
        on="uniqueID",
        how="left"
    )

    pivot["prompt"] = prompt

    wide_rows.append(pivot)


wide_df = pd.concat(
    wide_rows,
    ignore_index=True
)


# ============================================================
# 16. GENERALIZED JSD FOR THREE DISTRIBUTIONS
# ============================================================

def entropy_base2(p):

    p = np.asarray(p, dtype=float)

    p = np.clip(
        p,
        1e-15,
        1.0
    )

    return -np.sum(
        p * np.log2(p)
    )


def generalized_jsd(distributions):

    """
    Generalized Jensen-Shannon divergence for N distributions.

    For three seeds:

        JSD(P1, P2, P3)
        =
        H(M) - mean(H(Pi))

    where:

        M = mean(P1, P2, P3)

    With base-2 entropy, JSD is bounded in [0,1].
    """

    distributions = np.asarray(
        distributions,
        dtype=float
    )

    M = np.mean(
        distributions,
        axis=0
    )

    h_m = entropy_base2(M)

    h_each = np.mean([
        entropy_base2(p)
        for p in distributions
    ])

    jsd = h_m - h_each

    # Numerical cleanup
    return float(
        np.clip(jsd, 0.0, 1.0)
    )


# ============================================================
# 17. CALCULATE PER-TWEET JSD
# ============================================================

jsd_rows = []

seed_names = [
    "seed_1",
    "seed_2",
    "seed_3",
    'seed_4'
]

for _, row in wide_df.iterrows():

    prompt = row["prompt"]

    distributions = []

    predictions = []
    correctness = []
    probabilities = []

    for seed in seed_names:

        pred = row[
            f"prediction_{seed}"
        ]

        prob = row[
            f"predicted_class_probability_{seed}"
        ]

        distribution = make_distribution(
            pred,
            prob
        )

        distributions.append(
            distribution
        )

        predictions.append(pred)

        correctness.append(
            row[f"correct_{seed}"]
        )

        probabilities.append(prob)

    distributions = np.asarray(
        distributions
    )

    jsd = generalized_jsd(
        distributions
    )

    consistency = 1.0 - jsd

    # Exact prediction agreement
    all_same_prediction = (
        len(set(predictions)) == 1
    )

    # Exact probability equality is much stricter
    probability_range = (
        np.max(probabilities)
        -
        np.min(probabilities)
    )

    # Mean correctness across seeds
    mean_correctness = np.mean(
        correctness
    )

    all_correct = (
        mean_correctness == 1.0
    )

    all_wrong = (
        mean_correctness == 0.0
    )

    jsd_rows.append({

        "uniqueID":
            row["uniqueID"],

        "trigger_event":
            row["trigger_event"],

        "gold":
            row["gold"],

        "prompt":
            prompt,

        "JSD":
            jsd,

        "consistency_1_minus_JSD":
            consistency,

        "all_three_predictions_same":
            int(all_same_prediction),

        "all_three_correct":
            int(all_correct),

        "all_three_wrong":
            int(all_wrong),

        "mean_correctness_across_seeds":
            mean_correctness,

        "probability_range_across_seeds":
            probability_range,

        "seed1_prediction":
            predictions[0],

        "seed2_prediction":
            predictions[1],

        "seed3_prediction":
            predictions[2],

        "seed1_probability":
            probabilities[0],

        "seed2_probability":
            probabilities[1],

        "seed3_probability":
            probabilities[2]
    })


jsd_df = pd.DataFrame(
    jsd_rows
)


jsd_df.to_csv(
    OUTPUT_DIR / "FM4_per_tweet_JSD.csv",
    index=False
)


# ============================================================
# 18. JSD SUMMARY BY PROMPT
# ============================================================

jsd_summary = (
    jsd_df
    .groupby("prompt")
    .agg(
        N=("JSD", "size"),

        mean_JSD=("JSD", "mean"),

        median_JSD=("JSD", "median"),

        SD_JSD=("JSD", "std"),

        variance_JSD=("JSD", "var"),

        min_JSD=("JSD", "min"),

        max_JSD=("JSD", "max"),

        mean_consistency=(
            "consistency_1_minus_JSD",
            "mean"
        ),

        median_consistency=(
            "consistency_1_minus_JSD",
            "median"
        ),

        exact_prediction_agreement=(
            "all_three_predictions_same",
            "mean"
        ),

        all_three_correct_rate=(
            "all_three_correct",
            "mean"
        ),

        all_three_wrong_rate=(
            "all_three_wrong",
            "mean"
        )
    )
    .reset_index()
)


jsd_summary.to_csv(
    OUTPUT_DIR / "FM4_JSD_summary.csv",
    index=False
)

print("\nJSD summary:")
print(
    jsd_summary.to_string(index=False)
)


# ============================================================
# 19. STABLE-BUT-WRONG ANALYSIS
# ============================================================
#
# This is potentially one of the most useful FM4 diagnostics.
#
# We ask:
#
#     Among tweets for which all three seeds produce exactly
#     the same label, how often is that stable prediction wrong?
#
# This directly separates:
#
#     stability
#
# from:
#
#     correctness
#
# ============================================================

stable_rows = []

for prompt in PROMPTS:

    sub = jsd_df[
        jsd_df["prompt"] == prompt
    ].copy()

    stable = sub[
        sub["all_three_predictions_same"] == 1
    ]

    stable_rows.append({

        "prompt":
            prompt,

        "N_total":
            len(sub),

        "N_stable":
            len(stable),

        "stable_rate":
            len(stable) / len(sub),

        "stable_correct_rate":
            stable["all_three_correct"].mean()
            if len(stable) > 0 else np.nan,

        "stable_wrong_rate":
            stable["all_three_wrong"].mean()
            if len(stable) > 0 else np.nan,

        "N_stable_wrong":
            stable["all_three_wrong"].sum()
            if len(stable) > 0 else 0
    })


stable_df = pd.DataFrame(
    stable_rows
)


stable_df.to_csv(
    OUTPUT_DIR / "FM4_stable_but_wrong.csv",
    index=False
)


print("\nStable-but-wrong analysis:")
print(
    stable_df.to_string(index=False)
)


# ============================================================
# 20. JSD VS CORRECTNESS
# ============================================================
#
# Does instability correlate with being wrong?
#
# This is exploratory because JSD may be highly concentrated
# near zero.
#
# ============================================================

jsd_correctness_rows = []

for prompt in PROMPTS:

    sub = jsd_df[
        jsd_df["prompt"] == prompt
    ].copy()

    # Correctness averaged across the three seeds
    rho, p = spearmanr(
        sub["JSD"],
        sub["mean_correctness_across_seeds"]
    )

    jsd_correctness_rows.append({

        "prompt":
            prompt,

        "N":
            len(sub),

        "Spearman_rho_JSD_correctness":
            rho,

        "p":
            p,

        "mean_JSD_correct":
            sub.loc[
                sub["mean_correctness_across_seeds"] == 1,
                "JSD"
            ].mean(),

        "mean_JSD_wrong":
            sub.loc[
                sub["mean_correctness_across_seeds"] == 0,
                "JSD"
            ].mean()
    })


jsd_correctness_df = pd.DataFrame(
    jsd_correctness_rows
)


jsd_correctness_df.to_csv(
    OUTPUT_DIR / "FM4_JSD_vs_correctness.csv",
    index=False
)


# ============================================================
# 21. JSD HIGH-VS-LOW STABILITY ANALYSIS
# ============================================================
#
# Split at the median JSD.
#
# This is descriptive, not a validated threshold.
#
# ============================================================

jsd_split_rows = []

for prompt in PROMPTS:

    sub = jsd_df[
        jsd_df["prompt"] == prompt
    ].copy()

    median_jsd = sub["JSD"].median()

    sub["JSD_group"] = np.where(
        sub["JSD"] <= median_jsd,
        "low_JSD",
        "high_JSD"
    )

    for group, g in sub.groupby("JSD_group"):

        jsd_split_rows.append({

            "prompt":
                prompt,

            "JSD_group":
                group,

            "N":
                len(g),

            "median_JSD":
                median_jsd,

            "mean_JSD":
                g["JSD"].mean(),

            "mean_correctness":
                g["mean_correctness_across_seeds"].mean(),

            "all_three_wrong_rate":
                g["all_three_wrong"].mean(),

            "all_three_same_rate":
                g["all_three_predictions_same"].mean()
        })


jsd_split_df = pd.DataFrame(
    jsd_split_rows
)

jsd_split_df.to_csv(
    OUTPUT_DIR / "FM4_JSD_low_high_split.csv",
    index=False
)


# ============================================================
# 22. SEED-TO-SEED AGREEMENT
# ============================================================
#
# Pairwise exact prediction agreement.
#
# ============================================================

agreement_rows = []

for prompt in PROMPTS:

    sub = wide_df[
        wide_df["prompt"] == prompt
    ].copy()

    pairs = [
        ("seed_1", "seed_2"),
        ("seed_1", "seed_3"),
        ("seed_2", "seed_3")
    ]

    for s1, s2 in pairs:

        agreement = (
            sub[f"prediction_{s1}"]
            ==
            sub[f"prediction_{s2}"]
        ).mean()

        agreement_rows.append({

            "prompt":
                prompt,

            "seed_pair":
                f"{s1}_vs_{s2}",

            "N":
                len(sub),

            "exact_label_agreement":
                agreement
        })


agreement_df = pd.DataFrame(
    agreement_rows
)

agreement_df.to_csv(
    OUTPUT_DIR / "FM4_seed_pairwise_agreement.csv",
    index=False
)


# ============================================================
# 23. PER-TWEET PROBABILITY VARIABILITY
# ============================================================
#
# Even when all three seeds predict the same label, their
# probabilities may differ.
#
# We therefore calculate:
#
#     mean probability
#     SD probability
#     range probability
#
# ============================================================

probability_variability_rows = []

for _, row in wide_df.iterrows():

    for probability_type in [
        "predicted_class_probability"
    ]:

        values = np.array([
            row[
                f"{probability_type}_{seed}"
            ]
            for seed in seed_names
        ])

        probability_variability_rows.append({

            "uniqueID":
                row["uniqueID"],

            "prompt":
                row["prompt"],

            "mean_probability":
                np.mean(values),

            "SD_probability":
                np.std(values, ddof=1),

            "range_probability":
                np.ptp(values),

            "min_probability":
                np.min(values),

            "max_probability":
                np.max(values)
        })


probability_variability_df = pd.DataFrame(
    probability_variability_rows
)


probability_variability_df.to_csv(
    OUTPUT_DIR / "FM4_probability_seed_variability.csv",
    index=False
)


# ============================================================
# 24. FIGURE 1:
#     PROBABILITY VS CORRECTNESS
# ============================================================
#
# One figure per prompt.
#
# Correct = 1
# Incorrect = 0
#
# Jitter is added only to make overlapping points visible.
#
# ============================================================

for prompt in PROMPTS:

    sub = long_df[
        long_df["prompt"] == prompt
    ].copy()

    rng_plot = np.random.default_rng(123)

    y = (
        sub["correct"].to_numpy()
        +
        rng_plot.normal(
            0,
            0.035,
            size=len(sub)
        )
    )

    plt.figure(
        figsize=(7, 4.5)
    )

    plt.scatter(
        sub["predicted_class_probability"],
        y,
        alpha=0.20,
        s=12
    )

    plt.yticks(
        [0, 1],
        ["Incorrect", "Correct"]
    )

    plt.xlabel(
        "Probability assigned to predicted class"
    )

    plt.ylabel(
        "Prediction correctness"
    )

    plt.title(
        f"FM4: Probability vs Correctness — {prompt}"
    )

    plt.xlim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        f"FM4_probability_correctness_{prompt}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 25. FIGURE 2:
#     CORRELATION ACROSS SEEDS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

x_positions = np.arange(
    len(PROMPTS)
)

for seed_number, seed in enumerate(
    sorted(long_df["seed"].unique())
):

    vals = []

    for prompt in PROMPTS:

        row = correlation_df[
            (correlation_df["seed"] == seed) &
            (correlation_df["prompt"] == prompt)
        ]

        vals.append(
            row["point_biserial_r"].iloc[0]
        )

    plt.plot(
        x_positions,
        vals,
        marker="o",
        label=seed
    )


plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xticks(
    x_positions,
    PROMPTS
)

plt.ylabel(
    "Point-biserial correlation (r)"
)

plt.xlabel(
    "Prompt condition"
)

plt.title(
    "FM4: Probability–Correctness Association Across Seeds"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "FM4_correlation_across_seeds.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 26. FIGURE 3:
#     JSD DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(8, 5)
)

data_for_boxplot = [
    jsd_df[
        jsd_df["prompt"] == prompt
    ]["JSD"].values
    for prompt in PROMPTS
]

plt.boxplot(
    data_for_boxplot,
    labels=PROMPTS,
    showfliers=False
)

plt.ylabel(
    "Jensen–Shannon divergence across seeds"
)

plt.xlabel(
    "Prompt condition"
)

plt.title(
    "FM4: Prediction Stability Across Random Seeds"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "FM4_JSD_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 27. FIGURE 4:
#     PROBABILITY-BIN CORRECTNESS
# ============================================================

for prompt in PROMPTS:

    sub = bins_df[
        bins_df["prompt"] == prompt
    ].copy()

    plt.figure(
        figsize=(7, 4.5)
    )

    for seed in sub["seed"].unique():

        g = sub[
            sub["seed"] == seed
        ].copy()

        g = g[
            g["N"] > 0
        ]

        plt.plot(
            g["mean_probability"],
            g["empirical_correctness"],
            marker="o",
            label=seed
        )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel(
        "Mean probability assigned to predicted class"
    )

    plt.ylabel(
        "Empirical correctness rate"
    )

    plt.title(
        f"Probability-bin diagnostic — {prompt}"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIG_DIR /
        f"FM4_probability_bins_{prompt}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 28. FIGURE 5:
#     STABILITY VS CORRECTNESS
# ============================================================

plt.figure(
    figsize=(8, 5)
)

for i, prompt in enumerate(PROMPTS):

    sub = jsd_df[
        jsd_df["prompt"] == prompt
    ]

    plt.scatter(
        sub["JSD"],
        sub["mean_correctness_across_seeds"]
        +
        np.random.default_rng(i).normal(
            0,
            0.025,
            size=len(sub)
        ),
        alpha=0.15,
        s=10,
        label=prompt
    )

plt.xlabel(
    "Jensen–Shannon divergence across seeds"
)

plt.ylabel(
    "Mean correctness across seeds"
)

plt.yticks(
    [0, 1],
    ["Incorrect", "Correct"]
)

plt.title(
    "FM4: Seed Stability and Correctness"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_DIR /
    "FM4_JSD_vs_correctness.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 29. PAPER-READY SUMMARY TABLE
# ============================================================
#
# This combines:
#
#   correlation
#   bootstrap CI
#   AUROC
#   JSD
#   agreement
#
# ============================================================

summary_table = (
    correlation_summary
    .merge(
        bootstrap_df
        .groupby("prompt")
        .agg(
            bootstrap_r=("r", "mean"),
            CI_lower=("bootstrap_CI_lower", "min"),
            CI_upper=("bootstrap_CI_upper", "max")
        )
        .reset_index(),
        on="prompt",
        how="left"
    )
    .merge(
        auc_df
        .groupby("prompt")
        .agg(
            mean_AUROC=(
                "AUROC_correctness",
                "mean"
            ),
            min_AUROC=(
                "AUROC_correctness",
                "min"
            ),
            max_AUROC=(
                "AUROC_correctness",
                "max"
            )
        )
        .reset_index(),
        on="prompt",
        how="left"
    )
    .merge(
        jsd_summary,
        on="prompt",
        how="left"
    )
)


summary_table.to_csv(
    OUTPUT_DIR /
    "FM4_paper_ready_summary.csv",
    index=False
)


# ============================================================
# 30. LATEX TABLE GENERATION
# ============================================================

latex_table = summary_table.to_latex(
    index=False,
    float_format=lambda x: f"{x:.3f}"
)

with open(
    OUTPUT_DIR / "FM4_paper_ready_summary.tex",
    "w",
    encoding="utf-8"
) as f:

    f.write(latex_table)


# ============================================================
# 31. SAVE IMPORTANT "STABLE BUT WRONG" EXAMPLES
# ============================================================
#
# These can be inspected manually and potentially used as
# qualitative examples in the paper.
#
# We do NOT automatically put examples into the paper.
#
# ============================================================

stable_wrong_examples = (
    jsd_df[
        (jsd_df["all_three_predictions_same"] == 1) &
        (jsd_df["all_three_wrong"] == 1)
    ]
    .sort_values(
        [
            "prompt",
            "JSD"
        ]
    )
)

stable_wrong_examples.to_csv(
    OUTPUT_DIR /
    "FM4_stable_wrong_examples.csv",
    index=False
)


# ============================================================
# 32. MOST CONFIDENT WRONG PREDICTIONS
# ============================================================
#
# For each prompt and seed, identify the highest probability
# predictions that were nevertheless incorrect.
#
# Again: inspect these before using them as qualitative examples.
#
# ============================================================

confident_wrong_rows = []

for seed in long_df["seed"].unique():

    for prompt in PROMPTS:

        sub = long_df[
            (long_df["seed"] == seed) &
            (long_df["prompt"] == prompt) &
            (long_df["correct"] == 0)
        ].copy()

        top = (
            sub
            .sort_values(
                "predicted_class_probability",
                ascending=False
            )
            .head(50)
        )

        top = top.copy()

        top["rank"] = np.arange(
            1,
            len(top) + 1
        )

        confident_wrong_rows.append(
            top
        )


confident_wrong_df = pd.concat(
    confident_wrong_rows,
    ignore_index=True
)


confident_wrong_df.to_csv(
    OUTPUT_DIR /
    "FM4_most_confident_wrong_predictions.csv",
    index=False
)


# ============================================================
# 33. MOST UNCERTAIN CORRECT PREDICTIONS
# ============================================================
#
# These are the correct predictions with the LOWEST probability
# assigned to the predicted class.
#
# They are useful as the opposite diagnostic:
#
#     low apparent confidence
#     + correct prediction
#
# ============================================================

uncertain_correct_rows = []

for seed in long_df["seed"].unique():

    for prompt in PROMPTS:

        sub = long_df[
            (long_df["seed"] == seed) &
            (long_df["prompt"] == prompt) &
            (long_df["correct"] == 1)
        ].copy()

        bottom = (
            sub
            .sort_values(
                "predicted_class_probability",
                ascending=True
            )
            .head(50)
        )

        bottom = bottom.copy()

        bottom["rank"] = np.arange(
            1,
            len(bottom) + 1
        )

        uncertain_correct_rows.append(
            bottom
        )


uncertain_correct_df = pd.concat(
    uncertain_correct_rows,
    ignore_index=True
)


uncertain_correct_df.to_csv(
    OUTPUT_DIR /
    "FM4_most_uncertain_correct_predictions.csv",
    index=False
)


# ============================================================
# 34. OVERALL SUMMARY PRINT-OUT
# ============================================================

print("\n")
print("=" * 80)
print("FM4 ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nOutput directory:\n"
    f"  {OUTPUT_DIR.resolve()}"
)

print("\nKey files:")

key_files = [

    "FM4_master_long.csv",

    "FM4_performance_by_seed_prompt.csv",

    "FM4_probability_correctness_correlations.csv",

    "FM4_correlation_summary_across_seeds.csv",

    "FM4_bootstrap_pointbiserial_CI.csv",

    "FM4_logistic_clustered.csv",

    "FM4_AUROC_correctness.csv",

    "FM4_probability_bins.csv",

    "FM4_per_tweet_JSD.csv",

    "FM4_JSD_summary.csv",

    "FM4_stable_but_wrong.csv",

    "FM4_JSD_vs_correctness.csv",

    "FM4_seed_pairwise_agreement.csv",

    "FM4_probability_seed_variability.csv",

    "FM4_paper_ready_summary.csv",

    "FM4_paper_ready_summary.tex",

    "FM4_stable_wrong_examples.csv",

    "FM4_most_confident_wrong_predictions.csv",

    "FM4_most_uncertain_correct_predictions.csv"
]

for f in key_files:

    print(
        "  ✓",
        OUTPUT_DIR / f
    )

print(
    "\nFigures:"
)

for f in FIG_DIR.glob("*.png"):

    print(
        "  ✓",
        f
    )

print("\nDone.")
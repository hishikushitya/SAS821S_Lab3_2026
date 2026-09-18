from pathlib import Path
import pandas as pd

# Locate the data beside this script.
DATA = Path(__file__).resolve().parent

# Load the supplied text datasets.
training = pd.read_csv(DATA / "egs_text_training.csv")
investigation = pd.read_csv(DATA / "egs_text_investigation.csv")

# Inspect each dataset without changing the source files.
for name, original in [
    ("TRAINING REPORTS", training),
    ("INVESTIGATION REPORTS", investigation),
]:
    # Treat blank or whitespace-only cells as missing.
    df = original.replace(r"^\s*$", pd.NA, regex=True)

    print(f"\n--- {name} ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nMissing values per column:")
    print(df.isna().sum().to_string())

    print("\nDuplicate complete rows:", df.duplicated().sum())
    print("Duplicate report IDs:", df["report_id"].duplicated().sum())
    print(
        "Repeated report texts beyond the first occurrence:",
        df["report_text"].dropna().duplicated().sum(),
    )

    dates = pd.to_datetime(df["published_at"], errors="coerce")
    print("\nEarliest publication:", dates.min())
    print("Latest publication:", dates.max())
    print("Missing or invalid publication dates:", dates.isna().sum())

    print("\nReports by source:")
    print(df["source"].value_counts(dropna=False).to_string())

print("\n--- TRAINING CLASS BALANCE ---")
counts = training["relevant_label"].value_counts().sort_index()
balance = pd.DataFrame({
    "report_count": counts,
    "percentage": (counts / len(training) * 100).round(2),
})
print(balance.to_string())

# B1: TEXT PREPARATION AND QUALITY CHECKS
import re
import unicodedata

OUTPUT = DATA.parent / "outputs"
OUTPUT.mkdir(exist_ok=True)


def normalise_text(value):
    """Standardise text while preserving punctuation and indicators."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


# Add a prepared column; retain the original report text.
for df in (training, investigation):
    df["text_clean"] = df["report_text"].apply(normalise_text)

# Identify identical texts with more than one historical label.
text_groups = training.groupby("text_clean").agg(
    report_count=("report_id", "size"),
    distinct_labels=("relevant_label", "nunique"),
)

conflicting_texts = text_groups.index[
    text_groups["distinct_labels"] > 1
]

training["label_conflict"] = training["text_clean"].isin(
    conflicting_texts
)

# Check whether investigation text has appeared in training.
investigation["seen_in_training"] = investigation["text_clean"].isin(
    training["text_clean"]
)

print("\n--- TEXT QUALITY CHECKS ---")
print("Unique training texts:", training["text_clean"].nunique())
print("Unique investigation texts:", investigation["text_clean"].nunique())
print("Text groups with conflicting labels:", len(conflicting_texts))
print("Training rows in those groups:", training["label_conflict"].sum())
print(
    "Investigation rows matching training text:",
    investigation["seen_in_training"].sum(),
)

# Export conflicting records for review, without deleting them.
training.loc[
    training["label_conflict"],
    ["report_id", "report_text", "relevant_label", "text_clean"],
].to_csv(OUTPUT / "B1_label_conflicts.csv", index=False)

# Summary table 1: training class balance.
balance.index.name = "relevant_label"
balance.to_csv(OUTPUT / "B1_class_balance.csv")

# Summary table 2: sources in both datasets.
sources = pd.concat(
    [
        training["source"].value_counts().rename("training_reports"),
        investigation["source"].value_counts().rename("investigation_reports"),
    ],
    axis=1,
).fillna(0).astype(int)

sources.index.name = "source"
sources.to_csv(OUTPUT / "B1_source_distribution.csv")

# Summary table 3: publication patterns by month.
for name, df in [("training", training), ("investigation", investigation)]:
    dates = pd.to_datetime(df["published_at"], errors="coerce")
    monthly = dates.dt.to_period("M").value_counts().sort_index()
    monthly.index.name = "publication_month"
    monthly.rename("report_count").to_csv(
        OUTPUT / f"B1_{name}_monthly_counts.csv"
    )

print("\nB1 audit and summary tables saved in:", OUTPUT)


# B2: EXTRACT INDICATORS AND NAMED ENTITIES
import ipaddress

# Transparent dictionaries based on names in the supplied reports.
ACTORS = [
    "Kalahari Jackal",
    "Copper Dune",
    "Night Heron",
    "Red Kudu",
]

MALWARE = [
    "Sandstorm Loader",
    "BreakerFox",
    "DustRAT",
    "GridLock",
]

DOMAIN_PATTERN = (
    r"(?<![\w.-])"
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z]{2,63}\b"
)


def extract_entities(text):
    found = set()

    # Find IPv4 candidates, then validate their numerical ranges.
    for value in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
        try:
            address = str(ipaddress.IPv4Address(value))
            found.add(("ipv4", address))
        except ipaddress.AddressValueError:
            continue

    # Extract domain names, including the lab's .example domains.
    for value in re.findall(DOMAIN_PATTERN, text, flags=re.I):
        found.add(("domain", value.lower()))

    # Classify full-length hexadecimal hash candidates by length.
    hash_pattern = r"\b(?:[a-f0-9]{64}|[a-f0-9]{40}|[a-f0-9]{32})\b"
    for value in re.findall(hash_pattern, text, flags=re.I):
        hash_type = {32: "md5", 40: "sha1", 64: "sha256"}[len(value)]
        found.add((hash_type, value.lower()))

    # Keep explicitly labelled short hashes separate.
    fragment_pattern = r"\bhash(?:\s+is)?\s+([a-f0-9]{8,63})\b"
    for value in re.findall(fragment_pattern, text, flags=re.I):
        if len(value) not in (32, 40):
            found.add(("hash_fragment", value.lower()))

    # Match known names; this is dictionary matching, not attribution.
    for entity_type, names in [
        ("threat_actor", ACTORS),
        ("malware", MALWARE),
    ]:
        for name in names:
            pattern = r"\b" + re.escape(name) + r"\b"
            if re.search(pattern, text, flags=re.I):
                found.add((entity_type, name))

    return sorted(found)


extracted_rows = []

for row in investigation.itertuples(index=False):
    # Use original text so punctuation and indicator spelling survive.
    text = f"{row.title} {row.report_text}"

    for entity_type, entity_value in extract_entities(text):
        extracted_rows.append({
            "report_id": row.report_id,
            "entity_type": entity_type,
            "entity_value": entity_value,
            "analytical_relevance": (
                "Extracted mention; feed matching and internal "
                "corroboration pending."
            ),
        })

extracted = pd.DataFrame(
    extracted_rows,
    columns=[
        "report_id",
        "entity_type",
        "entity_value",
        "analytical_relevance",
    ],
)

extracted.to_csv(OUTPUT / "extracted_iocs.csv", index=False)

print("\n--- EXTRACTED ENTITIES ---")
print(extracted["entity_type"].value_counts().to_string())
print("Total extracted rows:", len(extracted))
print("Reports containing extracted entities:", extracted["report_id"].nunique())
print("Saved:", OUTPUT / "extracted_iocs.csv")

# B2: Match extracted indicators to the supplied IOC feed.
feed = pd.read_csv(DATA / "egs_ioc_feed.csv")

extracted["match_value"] = (
    extracted["entity_value"].str.strip().str.lower()
)
feed["match_value"] = (
    feed["indicator_value"].str.strip().str.lower()
)

enriched = extracted.merge(
    feed[
        [
            "indicator_type", "match_value", "ioc_id",
            "confidence", "context", "first_seen",
        ]
    ],
    left_on=["entity_type", "match_value"],
    right_on=["indicator_type", "match_value"],
    how="left",
    validate="many_to_one",
)

matched = enriched.loc[enriched["ioc_id"].notna()]

matched.to_csv(OUTPUT / "B2_feed_matches.csv", index=False)

print("\n--- IOC FEED MATCHES ---")
print("Extracted rows matching the feed:", len(matched))
print("Distinct matched IOC IDs:", matched["ioc_id"].nunique())
print(
    matched[
        ["report_id", "entity_type", "ioc_id", "confidence"]
    ].to_string(index=False)
)

# B2: Add dates, internal evidence and analytical relevance.
events = pd.read_csv(DATA / "egs_security_event_logs.csv")
events["match_value"] = (
    events["indicator"].fillna("").str.strip().str.lower()
)

enriched = enriched.merge(
    investigation[["report_id", "published_at"]],
    on="report_id",
    how="left",
    validate="many_to_one",
)

# Use the fictional scenario's assessment date.
assessment_date = pd.Timestamp("2026-10-19")
published = pd.to_datetime(enriched["published_at"], errors="coerce")
first_seen = pd.to_datetime(enriched["first_seen"], errors="coerce")

enriched["report_age_days"] = (assessment_date - published).dt.days
enriched["indicator_age_days"] = (assessment_date - first_seen).dt.days
enriched["report_predates_first_seen"] = published < first_seen

enriched["internal_event_ids"] = ""
enriched["internal_actions"] = ""
enriched["internal_event_count"] = 0

for index, row in enriched.iterrows():
    if pd.notna(row["ioc_id"]):
        hits = events.loc[
            events["match_value"] == row["match_value"]
        ]

        enriched.at[index, "internal_event_ids"] = ";".join(
            hits["event_id"].astype(str)
        )
        enriched.at[index, "internal_actions"] = ";".join(
            sorted(hits["action"].dropna().unique())
        )
        enriched.at[index, "internal_event_count"] = len(hits)

        note = (
            f"Feed match {row['ioc_id']}; "
            f"confidence={row['confidence']}; "
            f"context={row['context']}. "
        )

        if len(hits) > 0:
            note += (
                f"{len(hits)} exact matches in the event indicator field. "
                "Review actions and context before concluding compromise."
            )
        else:
            note += (
                "No exact match in the event indicator field; "
                "this does not prove absence from all telemetry."
            )

        if row["report_predates_first_seen"]:
            note += (
                " Publication precedes feed first_seen; "
                "investigate the date discrepancy."
            )

    elif row["entity_type"] == "hash_fragment":
        note = "Incomplete hash; cannot establish a full-hash match."

    elif row["entity_type"] in ("threat_actor", "malware"):
        note = (
            "Named entity mentioned in a report; "
            "not proof of attribution or infection."
        )

    else:
        note = (
            "No exact match in the supplied feed; "
            "this does not establish whether the entity is benign."
        )

    enriched.at[index, "analytical_relevance"] = note

# Save the enriched version of the required extraction file.
export_columns = [
    column for column in enriched.columns
    if column not in ("match_value", "indicator_type")
]
enriched[export_columns].to_csv(
    OUTPUT / "extracted_iocs.csv", index=False
)

matched = enriched.loc[enriched["ioc_id"].notna()]
matched[export_columns].to_csv(
    OUTPUT / "B2_feed_matches.csv", index=False
)

print("\n--- B2 INTERNAL CORRELATION ---")
print(
    matched[
        ["report_id", "ioc_id", "internal_event_count",
         "internal_event_ids"]
    ].to_string(index=False)
)
print(
    "Matched rows with publication before feed first_seen:",
    int(matched["report_predates_first_seen"].sum()),
)
print("Updated:", OUTPUT / "extracted_iocs.csv")

# B3: TEXT CLASSIFICATION AND EVALUATION
import sklearn
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

SEED = 8213026

# Only report text is used as a model input.
X = training["text_clean"].reset_index(drop=True)
y = training["relevant_label"].astype(int).reset_index(drop=True)


def make_text_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=None,
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])


# Required evaluation: 80% training, 20% testing.
baseline_train, baseline_test = train_test_split(
    training.index.to_numpy(),
    test_size=0.20,
    stratify=y,
    random_state=SEED,
)

# Additional evaluation: identical texts stay in the same group.
group_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)
group_train, group_test = next(
    group_splitter.split(X, y, groups=X)
)

evaluation_rows = []

for name, train_ids, test_ids in [
    ("stratified_baseline", baseline_train, baseline_test),
    ("grouped_check", group_train, group_test),
]:
    text_model = make_text_model()

    # Fit vocabulary, IDF weights and classifier on training rows only.
    text_model.fit(X.iloc[train_ids], y.iloc[train_ids])
    predicted = text_model.predict(X.iloc[test_ids])
    actual = y.iloc[test_ids]

    cm = confusion_matrix(actual, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    shared_texts = len(
        set(X.iloc[train_ids]) & set(X.iloc[test_ids])
    )

    metrics = {
        "evaluation": name,
        "train_rows": len(train_ids),
        "test_rows": len(test_ids),
        "shared_texts": shared_texts,
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "false_negative_rate": fn / (fn + tp) if fn + tp else float("nan"),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    evaluation_rows.append(metrics)

    print(f"\n--- B3: {name.upper()} ---")
    print("Training rows:", len(train_ids))
    print("Test rows:", len(test_ids))
    print("Identical texts shared across the split:", shared_texts)
    print("\nConfusion matrix:")
    print(pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_string())

    for metric in [
        "accuracy", "precision", "recall", "f1", "false_negative_rate"
    ]:
        print(f"{metric}: {metrics[metric]:.4f}")

    pd.DataFrame({
        "report_id": training.iloc[test_ids]["report_id"].to_numpy(),
        "actual_label": actual.to_numpy(),
        "predicted_label": predicted,
    }).to_csv(OUTPUT / f"B3_{name}_predictions.csv", index=False)

pd.DataFrame(evaluation_rows).to_csv(
    OUTPUT / "B3_classifier_metrics.csv", index=False
)

print("\nscikit-learn version:", sklearn.__version__)
print("B3 evaluation files saved in:", OUTPUT)


# B3: Score investigation reports and export the top 15.
final_text_model = make_text_model()
final_text_model.fit(X, y)

positive_column = list(final_text_model.classes_).index(1)

scored_reports = investigation.copy()
scored_reports["relevance_probability"] = (
    final_text_model.predict_proba(scored_reports["text_clean"])[
        :, positive_column
    ]
)
scored_reports["predicted_relevant"] = (
    scored_reports["relevance_probability"] >= 0.5
).astype(int)
scored_reports["seen_in_training"] = (
    scored_reports["text_clean"].isin(X)
)

ranked_reports = scored_reports.sort_values(
    ["relevance_probability", "published_at", "report_id"],
    ascending=[False, False, True],
).copy()

ranked_reports.insert(
    0, "model_rank", range(1, len(ranked_reports) + 1)
)

ranked_reports.to_csv(
    OUTPUT / "B3_all_scored_reports.csv", index=False
)
ranked_reports.head(15).to_csv(
    OUTPUT / "top_15_relevant_reports.csv", index=False
)

print("\n--- TOP 15 REPORTS BY MODEL SCORE ---")
print(
    ranked_reports.head(15)[
        ["model_rank", "report_id", "published_at",
         "relevance_probability"]
    ].to_string(index=False)
)
print("Saved:", OUTPUT / "top_15_relevant_reports.csv")


# B3: Compare original and modified threat statements.
adversarial_text = pd.read_csv(
    DATA / "egs_adversarial_text_cases.csv"
)

for version in ("original", "modified"):
    prepared = adversarial_text[f"{version}_text"].apply(
        normalise_text
    )

    adversarial_text[f"{version}_probability"] = (
        final_text_model.predict_proba(prepared)[:, positive_column]
    )

    adversarial_text[f"{version}_prediction"] = (
        adversarial_text[f"{version}_probability"] >= 0.5
    ).astype(int)

# Modified score minus original score, in percentage points.
adversarial_text["change_percentage_points"] = 100 * (
    adversarial_text["modified_probability"]
    - adversarial_text["original_probability"]
)

adversarial_text["relevant_to_irrelevant_flip"] = (
    (adversarial_text["original_prediction"] == 1)
    & (adversarial_text["modified_prediction"] == 0)
)

adversarial_text.to_csv(
    OUTPUT / "B3_adversarial_text_results.csv",
    index=False,
)

print("\n--- ADVERSARIAL TEXT RESULTS ---")
print(
    adversarial_text[
        [
            "case_id",
            "original_probability",
            "modified_probability",
            "change_percentage_points",
            "relevant_to_irrelevant_flip",
        ]
    ].round(4).to_string(index=False)
)

print(
    "Relevant-to-irrelevant flips:",
    int(adversarial_text["relevant_to_irrelevant_flip"].sum()),
)
print("Saved:", OUTPUT / "B3_adversarial_text_results.csv")


# C1: BUILD AND CHECK THE DIRECTED NETWORK
import networkx as nx

assets = pd.read_csv(DATA / "egs_asset_inventory.csv")
edges = pd.read_csv(DATA / "egs_network_topology_edges.csv")
posture = pd.read_csv(DATA / "egs_host_security_posture.csv")
scenarios = pd.read_csv(DATA / "egs_simulation_scenarios.csv")

# Check identifiers before combining the evidence.
assert assets["asset_id"].is_unique, "Duplicate asset IDs"
assert posture["asset_id"].is_unique, "Duplicate posture IDs"
assert not edges.duplicated(
    ["source_asset", "target_asset"]
).any(), "Duplicate directed edges"

asset_ids = set(assets["asset_id"])
edge_ids = set(edges["source_asset"]) | set(edges["target_asset"])

assert edge_ids <= asset_ids, "An edge references an unknown asset"
assert asset_ids <= set(posture["asset_id"]), "Missing host posture"

assert edges["base_transmission_probability"].between(0, 1).all()
assert posture["susceptibility"].between(0, 1).all()

# Combine asset details and host security posture.
asset_details = assets.merge(
    posture,
    on="asset_id",
    how="left",
    validate="one_to_one",
)

G = nx.DiGraph()

# Add every asset, including assets with no listed connections.
for record in asset_details.to_dict("records"):
    asset_id = record.pop("asset_id")
    G.add_node(asset_id, **record)

# Connections retain their supplied direction and attributes.
for record in edges.to_dict("records"):
    source = record.pop("source_asset")
    target = record.pop("target_asset")
    G.add_edge(source, target, **record)

INITIAL_INFECTED = "VENDOR-LT-07"
assert INITIAL_INFECTED in G, "Initial infected asset is missing"

# Define outcomes explicitly using inventory classifications.
OT_NODES = {
    node for node, details in G.nodes(data=True)
    if details["zone"] == "OT"
}
SAFETY_NODES = {
    node for node, details in G.nodes(data=True)
    if details["zone"] == "Safety"
}
CRITICAL_NODES = {
    node for node, details in G.nodes(data=True)
    if details["criticality"] == "Critical"
}

reachable = nx.descendants(G, INITIAL_INFECTED)
unreachable = set(G.nodes) - reachable - {INITIAL_INFECTED}

print("\n--- C1 NETWORK CHECK ---")
print("Assets:", G.number_of_nodes())
print("Directed connections:", G.number_of_edges())
print("Initial infected asset:", INITIAL_INFECTED)
print("OT-zone assets:", len(OT_NODES))
print("Safety-zone assets:", len(SAFETY_NODES))
print("Critical assets across all zones:", len(CRITICAL_NODES))
print("Other assets reachable through listed paths:", len(reachable))
print("Unreachable from the initial asset:", ", ".join(sorted(unreachable)))

print("\nSupplied scenarios:")
print(
    scenarios[["scenario_id", "scenario_name"]].to_string(index=False)
)


# C2: MONTE CARLO PROPAGATION SIMULATION
import numpy as np

N_RUNS = 1000
MAX_STEPS = 12

PATCH_TARGETS = {
    "VENDOR-LT-07", "VPN-GW-01", "VENDOR-DMZ-01"
}

susceptibility = posture.set_index("asset_id")["susceptibility"].to_dict()
zone = assets.set_index("asset_id")["zone"].to_dict()

edr_windows = set(posture.loc[
    posture["operating_system"].str.contains(
        "Windows", case=False, na=False
    ) & posture["edr_present"].eq("Yes"),
    "asset_id"
])

# Fixed edge order makes the random experiments reproducible.
simulation_edges = sorted(
    (row.source_asset, row.target_asset,
     row.base_transmission_probability)
    for row in edges.itertuples(index=False)
)


def simulate_one(scenario, run_number,
                 max_steps=MAX_STEPS, probability_scale=1.0):
    # Reuse each trial's random sequence across control scenarios.
    rng = np.random.default_rng(
        np.random.SeedSequence([SEED, run_number])
    )

    infection_time = {INITIAL_INFECTED: 0}
    time_to_ot = np.nan

    for step in range(1, max_steps + 1):
        newly_infected = set()
        draws = rng.random(len(simulation_edges))

        for (source, target, base), draw in zip(simulation_edges, draws):
            if source not in infection_time or target in infection_time:
                continue

            probability = (
                base * susceptibility[target] * probability_scale
            )

            if target in PATCH_TARGETS:
                probability *= scenario.patch_factor

            if zone[target] in {"OT-DMZ", "OT"}:
                probability *= scenario.segmentation_factor

            if (
                source in edr_windows
                and step - infection_time[source]
                > scenario.isolation_delay_steps
            ):
                probability *= scenario.post_detection_transmission_factor

            if draw < np.clip(probability, 0.0, 1.0):
                newly_infected.add(target)

        # Update together, preventing multiple hops in one step.
        for target in sorted(newly_infected):
            infection_time[target] = step

        if np.isnan(time_to_ot) and newly_infected.intersection(OT_NODES):
            time_to_ot = step

    infected = set(infection_time)

    return {
        "reached_ot": bool(infected & OT_NODES),
        "reached_safety": bool(infected & SAFETY_NODES),
        "critical_infected": len(infected & CRITICAL_NODES),
        "total_infected": len(infected),
        "time_to_ot": time_to_ot,
    }


summaries = []

for scenario in scenarios.itertuples(index=False):
    runs = pd.DataFrame([
        simulate_one(scenario, run_number)
        for run_number in range(N_RUNS)
    ])

    runs.index.name = "run_number"
    runs.to_csv(OUTPUT / f"C2_runs_{scenario.scenario_id}.csv")

    successful_ot_times = runs["time_to_ot"].dropna()

    summaries.append({
        "scenario_id": scenario.scenario_id,
        "scenario_name": scenario.scenario_name,
        "iterations": N_RUNS,
        "max_steps": MAX_STEPS,
        "prob_reach_ot": runs["reached_ot"].mean(),
        "prob_reach_safety": runs["reached_safety"].mean(),
        "mean_critical_infected": runs["critical_infected"].mean(),
        "mean_total_infected": runs["total_infected"].mean(),
        "median_time_to_ot": successful_ot_times.median(),
        "p95_total_infected": runs["total_infected"].quantile(
            0.95, interpolation="higher"
        ),
    })

simulation_summary = pd.DataFrame(summaries)
baseline = simulation_summary.loc[
    simulation_summary["scenario_id"] == "S0"
].iloc[0]

# Compare both OT and safety outcomes with current controls.
for metric in ("prob_reach_ot", "prob_reach_safety"):
    reduction = baseline[metric] - simulation_summary[metric]

    simulation_summary[f"{metric}_absolute_reduction_pp"] = (
        100 * reduction
    )
    simulation_summary[f"{metric}_relative_reduction_pct"] = (
        100 * reduction / baseline[metric]
        if baseline[metric] > 0 else np.nan
    )

simulation_summary.to_csv(
    OUTPUT / "simulation_summary.csv", index=False
)

print("\n--- C2 SIMULATION RESULTS ---")
print(
    simulation_summary[
        [
            "scenario_id", "prob_reach_ot", "prob_reach_safety",
            "mean_critical_infected", "mean_total_infected",
            "median_time_to_ot", "p95_total_infected",
        ]
    ].round(4).to_string(index=False)
)
print("Saved:", OUTPUT / "simulation_summary.csv")


# C2: CONFIDENCE INTERVALS AND COMPARISON FIGURES
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = DATA.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# Approximate 95% Wilson intervals for simulated proportions.
def wilson_interval(probability, number_of_trials):
    z = 1.96
    denominator = 1 + z**2 / number_of_trials
    centre = (
        probability + z**2 / (2 * number_of_trials)
    ) / denominator
    margin = z * np.sqrt(
        probability * (1 - probability) / number_of_trials
        + z**2 / (4 * number_of_trials**2)
    ) / denominator

    return (
        np.clip(centre - margin, 0, 1),
        np.clip(centre + margin, 0, 1),
    )


labels = [
    "S0\nCurrent\ncontrols",
    "S1\nPatching",
    "S2\nSegmentation",
    "S3\nRapid\nisolation",
    "S4\nCombined",
]
chart_data = simulation_summary.sort_values("scenario_id").copy()
positions = np.arange(len(chart_data))

# Figure 1: OT and safety reach, with separate labelled scales.
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

for ax, metric, title in [
    (axes[0], "prob_reach_ot", "Reaching the OT zone"),
    (axes[1], "prob_reach_safety", "Reaching the safety zone"),
]:
    probabilities = chart_data[metric].to_numpy()
    trials = chart_data["iterations"].to_numpy()
    lower, upper = wilson_interval(probabilities, trials)

    chart_data[f"{metric}_ci95_low"] = lower
    chart_data[f"{metric}_ci95_high"] = upper

    values = 100 * probabilities
    errors = np.maximum(
        100 * np.vstack([
            probabilities - lower,
            upper - probabilities,
        ]),
        0,
    )

    ax.bar(
        positions, values, yerr=errors, capsize=4,
        color="#2563EB", ecolor="#334155",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("Simulated probability within 12 steps (%)")
    ax.set_ylim(0, 100 if metric == "prob_reach_ot" else 7)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.2)

    offset = 1.5 if metric == "prob_reach_ot" else 0.15
    for position, value, high in zip(positions, values, upper):
        ax.text(
            position, 100 * high + offset,
            f"{value:.1f}%", ha="center", fontsize=9,
        )

fig.suptitle("Control comparison: simulated zone reach")
fig.text(
    0.5, 0.02,
    "1,000 trials per scenario; 12 steps; bars show estimates "
    "and whiskers show 95% Wilson intervals. Panel scales differ.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(FIGURES / "C2_reach_probabilities.png", dpi=200)
plt.close(fig)

# Figure 2: average and high-percentile spread.
fig, ax = plt.subplots(figsize=(10, 5.5))
width = 0.25

for offset, column, label, colour in [
    (-width, "mean_total_infected", "Mean total", "#2563EB"),
    (0, "mean_critical_infected", "Mean critical", "#D97706"),
    (width, "p95_total_infected", "95th percentile total", "#64748B"),
]:
    bars = ax.bar(
        positions + offset, chart_data[column],
        width=width, label=label, color=colour,
    )
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)

ax.set_xticks(positions)
ax.set_xticklabels(labels)
ax.set_ylabel("Number of infected assets")
ax.set_title("Control comparison: extent of simulated spread")
ax.set_ylim(0, chart_data["p95_total_infected"].max() + 4)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=0.2)
ax.legend(loc="upper right", fontsize=9)

fig.text(
    0.5, 0.02,
    "1,000 trials per scenario; 12 steps. "
    "Total counts include the initially infected vendor laptop.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIGURES / "C2_infection_counts.png", dpi=200)
plt.close(fig)

# Preserve existing metrics and add the confidence intervals.
simulation_summary = chart_data
simulation_summary.to_csv(
    OUTPUT / "simulation_summary.csv", index=False
)

print("\n--- C2 CHARTS SAVED ---")
print(FIGURES / "C2_reach_probabilities.png")
print(FIGURES / "C2_infection_counts.png")
print("simulation_summary.csv now includes 95% confidence intervals.")


# C3: SENSITIVITY TO PROBABILITIES AND SIMULATION LENGTH
sensitivity_settings = [
    ("Baseline", 12, 1.0),
    ("Lower probability", 12, 0.75),
    ("Higher probability", 12, 1.25),
    ("Shorter horizon", 6, 1.0),
    ("Longer horizon", 24, 1.0),
]

sensitivity_rows = []

for test_name, horizon, scale in sensitivity_settings:
    print(f"\nRunning sensitivity test: {test_name}", flush=True)

    for scenario in scenarios.itertuples(index=False):
        if test_name == "Baseline":
            # Reuse the original experiment.
            result = simulation_summary.loc[
                simulation_summary["scenario_id"] == scenario.scenario_id
            ].iloc[0]

            p_ot = result["prob_reach_ot"]
            p_safety = result["prob_reach_safety"]
            mean_total = result["mean_total_infected"]

        else:
            trials = pd.DataFrame([
                simulate_one(
                    scenario,
                    run_number,
                    max_steps=horizon,
                    probability_scale=scale,
                )
                for run_number in range(N_RUNS)
            ])

            p_ot = trials["reached_ot"].mean()
            p_safety = trials["reached_safety"].mean()
            mean_total = trials["total_infected"].mean()

        sensitivity_rows.append({
            "test": test_name,
            "scenario_id": scenario.scenario_id,
            "iterations": N_RUNS,
            "max_steps": horizon,
            "probability_scale": scale,
            "prob_reach_ot": p_ot,
            "prob_reach_safety": p_safety,
            "mean_total_infected": mean_total,
        })

sensitivity = pd.DataFrame(sensitivity_rows)

# Compare controls with S0 under the SAME sensitivity setting.
baseline_by_test = sensitivity.loc[
    sensitivity["scenario_id"] == "S0"
].set_index("test")["prob_reach_ot"]

reference = sensitivity["test"].map(baseline_by_test)
difference = reference - sensitivity["prob_reach_ot"]

sensitivity["ot_absolute_reduction_pp"] = 100 * difference
sensitivity["ot_relative_reduction_pct"] = (
    100 * difference / reference.replace(0, np.nan)
)

sensitivity.to_csv(
    OUTPUT / "C3_sensitivity_results.csv", index=False
)

test_order = [setting[0] for setting in sensitivity_settings]

print("\n--- C3 SENSITIVITY: OT REACH (%) ---")
print(
    sensitivity.pivot(
        index="test",
        columns="scenario_id",
        values="prob_reach_ot",
    ).reindex(test_order).mul(100).round(1).to_string()
)

print("\n--- C3 SENSITIVITY: SAFETY REACH (%) ---")
print(
    sensitivity.pivot(
        index="test",
        columns="scenario_id",
        values="prob_reach_safety",
    ).reindex(test_order).mul(100).round(1).to_string()
)

print("Saved:", OUTPUT / "C3_sensitivity_results.csv")


# D1: TRAIN AND EVALUATE THE DAILY-RISK MODEL
from sklearn.preprocessing import StandardScaler

risk_train = pd.read_csv(
    DATA / "egs_daily_risk_training.csv",
    parse_dates=["date"],
)

RISK_TARGET = "incident_within_7d"
RISK_FEATURES = [
    "failed_remote_logins",
    "new_external_ips",
    "suspicious_dns_queries",
    "critical_vulnerabilities",
    "mean_patch_age_days",
    "high_severity_edr_alerts",
    "ot_scan_attempts",
    "high_relevance_threat_reports",
    "vendor_access_sessions",
    "after_hours_admin_actions",
]

risk_X = risk_train[RISK_FEATURES]
risk_y = risk_train[RISK_TARGET].astype(int)

assert not risk_X.isna().any().any(), "Missing risk features"
assert risk_y.isin([0, 1]).all(), "Unexpected target labels"

# Reserve the test set before selecting a threshold.
risk_X_dev, risk_X_test, risk_y_dev, risk_y_test = train_test_split(
    risk_X, risk_y,
    test_size=0.20,
    stratify=risk_y,
    random_state=SEED,
)

# Split development data into fitting and validation portions.
risk_X_fit, risk_X_val, risk_y_fit, risk_y_val = train_test_split(
    risk_X_dev, risk_y_dev,
    test_size=0.25,
    stratify=risk_y_dev,
    random_state=SEED,
)

risk_model = Pipeline([
    ("scale", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )),
])

# Scaling and model fitting use only the fitting portion.
risk_model.fit(risk_X_fit, risk_y_fit)
risk_positive_column = list(risk_model.classes_).index(1)

validation_probabilities = risk_model.predict_proba(
    risk_X_val
)[:, risk_positive_column]

threshold_rows = []

for threshold in np.round(np.arange(0.05, 0.951, 0.05), 2):
    predictions = validation_probabilities >= threshold

    threshold_rows.append({
        "threshold": threshold,
        "recall": recall_score(
            risk_y_val, predictions, zero_division=0
        ),
        "precision": precision_score(
            risk_y_val, predictions, zero_division=0
        ),
        "alert_days": int(predictions.sum()),
    })

threshold_table = pd.DataFrame(threshold_rows)
threshold_table.to_csv(
    OUTPUT / "D1_threshold_validation.csv", index=False
)

eligible = threshold_table.loc[threshold_table["recall"] >= 0.90]

if eligible.empty:
    raise ValueError("No candidate threshold achieved 90% validation recall.")

selected = eligible.sort_values(
    ["precision", "threshold"], ascending=[False, False]
).iloc[0]

risk_threshold = float(selected["threshold"])

print("\n--- D1 RISK MODEL SETUP ---")
print("Fitting days:", len(risk_X_fit))
print("Validation days:", len(risk_X_val))
print("Test days:", len(risk_X_test))
print(f"Selected threshold: {risk_threshold:.2f}")
print(f"Validation recall: {selected['recall']:.4f}")
print(f"Validation precision: {selected['precision']:.4f}")

# Evaluate without changing the model or selected threshold.
risk_test_probabilities = risk_model.predict_proba(
    risk_X_test
)[:, risk_positive_column]

risk_evaluation_rows = []

for name, threshold in [
    ("default_0.50", 0.50),
    ("validation_selected", risk_threshold),
]:
    predictions = risk_test_probabilities >= threshold
    cm = confusion_matrix(risk_y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    risk_evaluation_rows.append({
        "evaluation": name,
        "threshold": threshold,
        "accuracy": accuracy_score(risk_y_test, predictions),
        "precision": precision_score(
            risk_y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            risk_y_test, predictions, zero_division=0
        ),
        "f1": f1_score(risk_y_test, predictions, zero_division=0),
        "false_negative_rate": fn / (fn + tp),
        "alert_days": int(predictions.sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

    print(f"\nConfusion matrix: {name}")
    print(pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    ).to_string())

risk_metrics = pd.DataFrame(risk_evaluation_rows)
risk_metrics.to_csv(
    OUTPUT / "D1_risk_model_metrics.csv", index=False
)

print("\n--- D1 TEST RESULTS ---")
print(risk_metrics.round(4).to_string(index=False))

# D2: SCORE INVESTIGATION DAYS AND PLOT THE TREND
import matplotlib.dates as mdates

risk_days = pd.read_csv(
    DATA / "egs_daily_risk_investigation.csv",
    parse_dates=["date"],
).sort_values("date").reset_index(drop=True)

assert risk_days["date"].is_unique, "Duplicate investigation dates"
assert not risk_days[RISK_FEATURES].isna().any().any()

risk_days["risk_probability"] = risk_model.predict_proba(
    risk_days[RISK_FEATURES]
)[:, risk_positive_column]

risk_days["alert"] = (
    risk_days["risk_probability"] >= risk_threshold
)

# Daily totals may only be available after that day has ended.
# This is an explicit availability assumption, not supplied metadata.
risk_days["assumed_available_at"] = (
    risk_days["date"] + pd.Timedelta(days=1)
)

# Flag values outside the ranges used to FIT the model.
fit_minimum = risk_X_fit.min()
fit_maximum = risk_X_fit.max()

outside_range = (
    risk_days[RISK_FEATURES].lt(fit_minimum, axis="columns")
    | risk_days[RISK_FEATURES].gt(fit_maximum, axis="columns")
)

risk_days["outside_fit_range_count"] = outside_range.sum(axis=1)
risk_days["outside_fit_range_features"] = outside_range.apply(
    lambda row: ";".join(row.index[row]),
    axis=1,
)

risk_days.to_csv(
    OUTPUT / "D2_all_scored_days.csv", index=False
)

top_risk_days = risk_days.sort_values(
    ["risk_probability", "date"],
    ascending=[False, False],
).head(10).copy()

top_risk_days.insert(0, "risk_rank", range(1, len(top_risk_days) + 1))
top_risk_days.to_csv(
    OUTPUT / "top_10_risk_days.csv", index=False
)

print("\n--- D2 TOP 10 RISK DAYS ---")
print(
    top_risk_days[
        ["risk_rank", "date", "risk_probability",
         "alert", "outside_fit_range_count"]
    ].to_string(index=False)
)

alert_days = risk_days.loc[risk_days["alert"]]

print("\n--- D2 EARLIEST MODEL ALERT ---")
if alert_days.empty:
    print("No investigation day crossed the selected threshold.")
else:
    first_alert = alert_days.iloc[0]
    print("Indicator date:", first_alert["date"].date())
    print(
        "Assumed available from:",
        first_alert["assumed_available_at"].date(),
    )
    print(f"Model score: {first_alert['risk_probability']:.4f}")
    print(f"Selected threshold: {risk_threshold:.2f}")

# Plot scores by the date of the daily indicators.
fig, ax = plt.subplots(figsize=(11, 5.5))

ax.plot(
    risk_days["date"], risk_days["risk_probability"],
    marker="o", markersize=4, color="#2563EB",
    label="Daily model score",
)
ax.axhline(
    risk_threshold, color="#B45309", linestyle="--",
    label=f"Alert threshold: {risk_threshold:.2f}",
)
ax.axvline(
    pd.Timestamp("2026-10-17 20:18:02"),
    color="#475569", linestyle=":",
    label="Blocked Modbus write: EVT01586",
)

ax.set_title("Investigation-period risk scores")
ax.set_xlabel("Indicator date, Namibia local time")
ax.set_ylabel("Uncalibrated model score")
ax.set_ylim(0, 1.05)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.grid(alpha=0.2)
ax.legend(loc="upper left", fontsize=9)

fig.text(
    0.5, 0.02,
    "Retrospective scoring. Daily totals are assumed available the "
    "following day; scores are not confirmed incident probabilities.",
    ha="center", fontsize=9,
)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIGURES / "D2_risk_over_time.png", dpi=200)
plt.close(fig)

print("\nSaved:", OUTPUT / "top_10_risk_days.csv")
print("Saved:", FIGURES / "D2_risk_over_time.png")

# D2: BUILD A TRACEABLE EVIDENCE TIMELINE

# Events selected after reviewing the vendor-to-OT sequence.
selected_event_ids = [
    "EVT01574", "EVT01576", "EVT01577", "EVT01578",
    "EVT01580", "EVT01581", "EVT01582", "EVT01583",
    "EVT01586", "EVT01587", "EVT01588", "EVT01589",
    "EVT01590", "EVT01591",
]

selected_report_ids = [
    "RPT9001", "RPT9002", "RPT9003",
    "RPT9004", "RPT9005", "RPT9006",
]

assert set(selected_event_ids) <= set(events["event_id"])
assert set(selected_report_ids) <= set(investigation["report_id"])

selected_events = events.loc[
    events["event_id"].isin(selected_event_ids)
].copy()

selected_reports = investigation.loc[
    investigation["report_id"].isin(selected_report_ids)
].copy()

# Attach feed references already established in B2.
report_ioc_links = (
    matched.groupby("report_id")["ioc_id"]
    .apply(lambda values: ";".join(sorted(set(values))))
    .to_dict()
)

timeline_rows = []

# Add the derived model warning with its availability assumption.
if not alert_days.empty:
    first_alert = alert_days.iloc[0]

    timeline_rows.append({
        "timestamp": first_alert["assumed_available_at"],
        "time_basis": "Assumed next-day availability",
        "evidence_type": "Derived model warning",
        "reference_id": "D2_all_scored_days.csv",
        "summary": (
            f"Indicators dated {first_alert['date'].date()}; "
            f"score={first_alert['risk_probability']:.4f}; "
            f"threshold={risk_threshold:.2f}"
        ),
        "action": "Analyst review proposed",
        "source_asset": "",
        "destination_asset": "",
        "linked_ioc_ids": "",
    })

for row in selected_reports.itertuples(index=False):
    timeline_rows.append({
        "timestamp": pd.Timestamp(row.published_at),
        "time_basis": "Publication date only; receipt time unknown",
        "evidence_type": "Threat report",
        "reference_id": row.report_id,
        "summary": row.title,
        "action": "",
        "source_asset": "",
        "destination_asset": "",
        "linked_ioc_ids": report_ioc_links.get(row.report_id, ""),
    })

for row in selected_events.itertuples(index=False):
    timeline_rows.append({
        "timestamp": pd.Timestamp(row.timestamp),
        "time_basis": "Recorded event time, UTC+2",
        "evidence_type": "Internal event",
        "reference_id": row.event_id,
        "summary": f"{row.event_type}: {row.details}",
        "action": row.action,
        "source_asset": row.source_asset,
        "destination_asset": row.destination_asset,
        "linked_ioc_ids": "",
    })

timeline = pd.DataFrame(timeline_rows).sort_values(
    ["timestamp", "reference_id"]
).reset_index(drop=True)

# Display date-only records without inventing an exact event time.
timeline["display_time"] = timeline.apply(
    lambda row: (
        row["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        if row["evidence_type"] == "Internal event"
        else row["timestamp"].strftime("%Y-%m-%d")
    ),
    axis=1,
)

timeline.to_csv(
    OUTPUT / "D2_evidence_timeline.csv", index=False
)

print("\n--- D2 TIMELINE CHECK ---")
print("Selected internal events:", len(selected_events))
print("Selected intelligence reports:", len(selected_reports))
print("Total timeline entries:", len(timeline))
print(
    timeline[
        ["display_time", "evidence_type", "reference_id", "action"]
    ].to_string(index=False)
)
print("Saved:", OUTPUT / "D2_evidence_timeline.csv")

# D3: ADVERSARIAL RISK-FEATURE TESTING
adversarial_risk = pd.read_csv(
    DATA / "egs_adversarial_risk_cases.csv"
)

assert not adversarial_risk[RISK_FEATURES].isna().any().any()

adversarial_risk["risk_probability"] = risk_model.predict_proba(
    adversarial_risk[RISK_FEATURES]
)[:, risk_positive_column]

adversarial_risk["alert"] = (
    adversarial_risk["risk_probability"] >= risk_threshold
)

baseline_mask = adversarial_risk["variant"].eq("baseline_attack")
assert baseline_mask.sum() == 1, "Expected one baseline attack case"

baseline_score = float(
    adversarial_risk.loc[baseline_mask, "risk_probability"].iloc[0]
)

# Negative values mean a lower score than the baseline attack.
adversarial_risk["change_from_baseline_pp"] = 100 * (
    adversarial_risk["risk_probability"] - baseline_score
)

adversarial_risk["evaded_at_threshold"] = (
    (baseline_score >= risk_threshold)
    & ~adversarial_risk["alert"]
)

# Identify feature values outside the model-fitting ranges.
adversarial_outside = (
    adversarial_risk[RISK_FEATURES].lt(
        risk_X_fit.min(), axis="columns"
    )
    | adversarial_risk[RISK_FEATURES].gt(
        risk_X_fit.max(), axis="columns"
    )
)

adversarial_risk["outside_fit_range_count"] = (
    adversarial_outside.sum(axis=1)
)

adversarial_risk.to_csv(
    OUTPUT / "D3_adversarial_risk_results.csv",
    index=False,
)

print("\n--- D3 ADVERSARIAL RISK RESULTS ---")
print(
    adversarial_risk[
        [
            "case_id",
            "variant",
            "risk_probability",
            "change_from_baseline_pp",
            "alert",
            "evaded_at_threshold",
            "outside_fit_range_count",
        ]
    ].round(6).to_string(index=False)
)

print(f"Operating threshold: {risk_threshold:.2f}")
print(
    "Variants falling below the threshold:",
    int(adversarial_risk.loc[
        ~baseline_mask, "evaded_at_threshold"
    ].sum()),
)
print("Saved:", OUTPUT / "D3_adversarial_risk_results.csv")

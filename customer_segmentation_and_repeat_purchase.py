"""
Maven Market: Customer Segmentation (RFM + K-Means) and Repeat-Purchase Prediction
===================================================================================

Business questions
  1. Who are our customers? -> segment them by Recency / Frequency / Monetary value.
  2. Who is likely to buy again in the next N days? -> predict it, so marketing can
     focus on the right customers (retain valuable ones, win back at-risk ones).

How it works (no data leakage)
  * Pick a cutoff date = last transaction date - horizon (default 90 days).
  * Features + segments are built ONLY from transactions on/before the cutoff.
  * Label = did the customer buy at least once in the `horizon` days AFTER the cutoff?

Usage
  python customer_segmentation_and_repeat_purchase.py --data-dir ./data     # real Maven Market CSVs
  python customer_segmentation_and_repeat_purchase.py --demo                # synthetic data, to test the pipeline
"""
import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------------
# Column names used from the Maven Market tables. If your CSVs use different names,
# change them here (and only here).
# ----------------------------------------------------------------------------------
COL_DATE = "transaction_date"
COL_CUSTOMER = "customer_id"
COL_PRODUCT = "product_id"
COL_STORE = "store_id"
COL_QTY = "quantity"
COL_PRICE = "product_retail_price"

MODEL_FEATURES = [
    "recency_days",
    "frequency",
    "monetary",
    "avg_spend_per_visit",
    "n_products",
    "n_stores",
    "tenure_days",
    "visits_last_90d",
]


# ----------------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------------
def _to_number(series: pd.Series) -> pd.Series:
    """Convert values like '$1,234.50' to numbers (no-op if already numeric)."""
    if series.dtype == object or str(series.dtype).startswith("str"):
        series = series.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(series, errors="coerce")


def _find_csvs(data_dir: Path, keyword: str) -> list:
    files = sorted(p for p in data_dir.rglob("*.csv") if keyword in p.name.lower())
    if not files:
        raise FileNotFoundError(
            f"No CSV with '{keyword}' in its name was found under {data_dir.resolve()}"
        )
    return files


def _require(df: pd.DataFrame, cols: list, table: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Table '{table}' is missing columns {missing}. Columns found: {list(df.columns)}. "
            "Update the COL_* constants at the top of the script."
        )


def load_real_data(data_dir: str):
    d = Path(data_dir)
    read = lambda f: pd.read_csv(f, encoding_errors="replace")
    tx = pd.concat([read(f) for f in _find_csvs(d, "transactions")], ignore_index=True)
    products = read(_find_csvs(d, "products")[0])
    customers = read(_find_csvs(d, "customers")[0])
    return tx, products, customers


def make_demo_data(seed: int = 42, n_customers: int = 1500):
    """Synthetic data with the SAME schema as Maven Market. For testing the pipeline only."""
    rng = np.random.default_rng(seed)
    start, n_days = pd.Timestamp("1997-01-01"), 729
    products = pd.DataFrame(
        {COL_PRODUCT: np.arange(1, 201), COL_PRICE: np.round(rng.uniform(0.8, 9.0, 200), 2)}
    )
    customers = pd.DataFrame({COL_CUSTOMER: np.arange(1, n_customers + 1)})

    kinds = rng.choice(["loyal", "casual", "churned"], n_customers, p=[0.2, 0.5, 0.3])
    cid, day, pid, qty, store = [], [], [], [], []
    for c, kind in zip(customers[COL_CUSTOMER], kinds):
        if kind == "loyal":
            rate, first, last = rng.uniform(1 / 14, 1 / 6), rng.integers(0, 200), n_days - 1
        elif kind == "casual":
            rate, first, last = rng.uniform(1 / 90, 1 / 30), rng.integers(0, 400), n_days - 1
        else:  # churned: buys for a while, then stops
            rate, first = rng.uniform(1 / 30, 1 / 10), rng.integers(0, 200)
            last = int(rng.integers(n_days // 3, n_days - 60))
        visits = np.flatnonzero(rng.random(max(last - first, 0)) < rate) + first
        if len(visits) == 0:
            continue
        items = rng.integers(1, 6, size=len(visits))  # items per visit
        total = items.sum()
        cid.append(np.full(total, c))
        day.append(np.repeat(visits, items))
        pid.append(rng.integers(1, 201, total))
        qty.append(rng.integers(1, 4, total))
        store.append(np.full(total, rng.integers(1, 25)))

    tx = pd.DataFrame(
        {
            COL_DATE: start + pd.to_timedelta(np.concatenate(day), unit="D"),
            COL_PRODUCT: np.concatenate(pid),
            COL_CUSTOMER: np.concatenate(cid),
            COL_STORE: np.concatenate(store),
            COL_QTY: np.concatenate(qty),
        }
    )
    return tx, products, customers


def prepare_transactions(tx: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    _require(tx, [COL_DATE, COL_CUSTOMER, COL_PRODUCT, COL_STORE, COL_QTY], "transactions")
    _require(products, [COL_PRODUCT, COL_PRICE], "products")

    tx = tx.copy()
    tx[COL_DATE] = pd.to_datetime(tx[COL_DATE], errors="coerce")
    products = products[[COL_PRODUCT, COL_PRICE]].copy()
    products[COL_PRICE] = _to_number(products[COL_PRICE])

    tx = tx.merge(products, on=COL_PRODUCT, how="left")
    tx["revenue"] = _to_number(tx[COL_QTY]) * tx[COL_PRICE]

    before = len(tx)
    tx = tx.dropna(subset=[COL_DATE, COL_CUSTOMER, "revenue"])
    dropped = before - len(tx)
    if dropped:
        print(f"[clean] dropped {dropped:,} rows with missing date / customer / price")
    return tx


# ----------------------------------------------------------------------------------
# 2. Features (RFM + a few extra behaviour signals), built only from pre-cutoff data
# ----------------------------------------------------------------------------------
def build_features(pre: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    g = pre.groupby(COL_CUSTOMER)
    feats = g.agg(
        first_date=(COL_DATE, "min"),
        last_date=(COL_DATE, "max"),
        frequency=(COL_DATE, "nunique"),  # number of distinct visit days
        monetary=("revenue", "sum"),
        n_products=(COL_PRODUCT, "nunique"),
        n_stores=(COL_STORE, "nunique"),
    )
    feats["recency_days"] = (cutoff - feats["last_date"]).dt.days
    feats["tenure_days"] = (cutoff - feats["first_date"]).dt.days
    feats["avg_spend_per_visit"] = feats["monetary"] / feats["frequency"]

    recent = pre[pre[COL_DATE] > cutoff - pd.Timedelta(days=90)]
    visits_90 = recent.groupby(COL_CUSTOMER)[COL_DATE].nunique()
    feats["visits_last_90d"] = visits_90.reindex(feats.index).fillna(0)
    return feats.drop(columns=["first_date", "last_date"])


def build_label(post: pd.DataFrame, index: pd.Index) -> pd.Series:
    buyers = set(post[COL_CUSTOMER].unique())
    return pd.Series(index.isin(buyers).astype(int), index=index, name="repeat_purchase")


# ----------------------------------------------------------------------------------
# 3. Segmentation (K-Means on log-scaled RFM)
# ----------------------------------------------------------------------------------
def segment_customers(feats: pd.DataFrame, k=None, seed: int = 42):
    X = np.log1p(feats[["recency_days", "frequency", "monetary"]])
    Xs = StandardScaler().fit_transform(X)

    scores = {}
    for kk in range(3, 7):  # 3-6 segments keeps the result easy to explain to business users
        labels = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit_predict(Xs)
        scores[kk] = silhouette_score(
            Xs, labels, sample_size=min(5000, len(Xs)), random_state=seed
        )
    best_k = k or max(scores, key=scores.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit_predict(Xs)
    return labels, scores, best_k


def name_segments(feats: pd.DataFrame, labels: np.ndarray) -> pd.Series:
    """Give each cluster a readable name based on where its customers rank overall
    (percentiles: 1.0 = best). The names are heuristics - always review them against
    the segment profile table before presenting them."""
    pct = pd.DataFrame(
        {
            "recent": 1 - feats["recency_days"].rank(pct=True),  # 1.0 = bought most recently
            "freq": feats["frequency"].rank(pct=True),
            "value": feats["monetary"].rank(pct=True),
        }
    )
    prof = pct.groupby(np.asarray(labels)).mean()
    names = {}
    for c, r in prof.iterrows():
        if r["freq"] >= 0.7 and r["value"] >= 0.7 and r["recent"] >= 0.5:
            base = "Champions"
        elif r["recent"] >= 0.5:
            base = "Recent, lighter buyers"
        elif r["freq"] >= 0.4:
            base = "At-risk (previously active)"
        else:
            base = "Dormant"
        names[c] = f"{base} (C{c})"
    return pd.Series(labels, index=feats.index).map(names).rename("segment")


# ----------------------------------------------------------------------------------
# 4. Repeat-purchase model
# ----------------------------------------------------------------------------------
def train_and_evaluate(feats: pd.DataFrame, y: pd.Series, seed: int = 42):
    X = feats[MODEL_FEATURES]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    # Baseline: "the more recently someone bought, the more likely they come back"
    auc = {"Naive baseline (recency only)": roc_auc_score(y_te, -X_te["recency_days"])}

    logit = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
    )
    logit.fit(np.log1p(X_tr), y_tr)
    auc["Logistic Regression"] = roc_auc_score(y_te, logit.predict_proba(np.log1p(X_te))[:, 1])

    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=seed, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    rf_proba = rf.predict_proba(X_te)[:, 1]
    auc["Random Forest"] = roc_auc_score(y_te, rf_proba)

    # Business-friendly metrics for the Random Forest
    pred = (rf_proba >= 0.5).astype(int)
    top = np.argsort(-rf_proba)[: max(1, len(rf_proba) // 10)]
    base_rate = float(y_te.mean())
    extra = {
        "base_rate_test": base_rate,
        "precision@0.5": precision_score(y_te, pred, zero_division=0),
        "recall@0.5": recall_score(y_te, pred, zero_division=0),
        "top_decile_hit_rate": float(y_te.to_numpy()[top].mean()),
    }
    extra["top_decile_lift"] = extra["top_decile_hit_rate"] / base_rate if base_rate else np.nan

    imp = permutation_importance(
        rf, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=seed, n_jobs=-1
    )
    importance = pd.Series(imp.importances_mean, index=MODEL_FEATURES).sort_values()
    return rf, auc, extra, importance


# ----------------------------------------------------------------------------------
# 5. Charts
# ----------------------------------------------------------------------------------
def make_charts(out, feats, seg, silhouettes, best_k, seg_summary, base_rate, importance):
    # (a) silhouette by k
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(list(silhouettes), list(silhouettes.values()), marker="o")
    ax.axvline(best_k, ls="--", c="grey")
    ax.set(xlabel="Number of segments (k)", ylabel="Silhouette score", title="Choosing k")
    fig.tight_layout()
    fig.savefig(out / "fig1_silhouette.png", dpi=150)
    plt.close(fig)

    # (b) segments on recency vs monetary
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in sorted(seg.unique()):
        m = seg == name
        ax.scatter(feats.loc[m, "recency_days"], feats.loc[m, "monetary"], s=10, alpha=0.6, label=name)
    ax.set(yscale="log", xlabel="Recency (days since last purchase)",
           ylabel="Monetary (total spend, log)", title="Customer segments")
    ax.legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(out / "fig2_segments.png", dpi=150)
    plt.close(fig)

    # (c) repeat rate by segment
    s = seg_summary.sort_values("repeat_rate")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(s.index, s["repeat_rate"] * 100)
    ax.axvline(base_rate * 100, ls="--", c="grey", label=f"Overall: {base_rate:.0%}")
    ax.set(xlabel="% of customers who bought again in the next window",
           title="Repeat-purchase rate by segment")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig3_repeat_rate_by_segment.png", dpi=150)
    plt.close(fig)

    # (d) permutation importance
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(importance.index, importance.values)
    ax.set(xlabel="Drop in ROC-AUC when the feature is shuffled",
           title="Which features drive repeat purchase?")
    fig.tight_layout()
    fig.savefig(out / "fig4_feature_importance.png", dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="folder with the Maven Market CSV files")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (pipeline test only)")
    ap.add_argument("--horizon", type=int, default=90, help="days to look ahead for repeat purchase")
    ap.add_argument("--k", type=int, default=None, help="force the number of segments")
    ap.add_argument("--out", default="outputs", help="output folder")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.demo:
        print("*** DEMO MODE: synthetic data - do NOT report these numbers as real results ***")
        tx, products, customers = make_demo_data(args.seed)
    else:
        tx, products, customers = load_real_data(args.data_dir)
    tx = prepare_transactions(tx, products)

    last_day = tx[COL_DATE].max()
    cutoff = last_day - pd.Timedelta(days=args.horizon)
    pre = tx[tx[COL_DATE] <= cutoff]
    post = tx[(tx[COL_DATE] > cutoff) & (tx[COL_DATE] <= last_day)]
    print(f"[data] {len(tx):,} transaction rows | {tx[COL_CUSTOMER].nunique():,} customers")
    print(f"[split] history: up to {cutoff.date()} | prediction window: {args.horizon} days to {last_day.date()}")

    feats = build_features(pre, cutoff)
    y = build_label(post, feats.index)
    print(f"[label] {len(feats):,} customers with history | repeat rate = {y.mean():.1%}")
    if y.nunique() < 2:
        raise SystemExit("Label has only one class - try a different --horizon.")

    # --- segmentation ---
    labels, silhouettes, best_k = segment_customers(feats, args.k, args.seed)
    seg = name_segments(feats, labels)
    print(f"[segments] silhouette by k: { {k: round(v, 3) for k, v in silhouettes.items()} } -> k = {best_k}")

    seg_summary = (
        feats.assign(segment=seg, repeat=y)
        .groupby("segment")
        .agg(
            customers=("frequency", "size"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            repeat_rate=("repeat", "mean"),
        )
        .round(3)
    )
    seg_summary["share_of_customers"] = (seg_summary["customers"] / len(feats)).round(3)
    seg_summary.to_csv(out / "segment_summary.csv")
    print("\n=== Segment profile ===")
    print(seg_summary.sort_values("avg_monetary", ascending=False).to_string())

    # --- prediction ---
    rf, auc, extra, importance = train_and_evaluate(feats, y, args.seed)
    print("\n=== Repeat-purchase model (held-out 25% of customers) ===")
    for name, v in auc.items():
        print(f"  ROC-AUC  {name:<32} {v:.3f}")
    print(f"  Base repeat rate (test)          {extra['base_rate_test']:.1%}")
    print(f"  Random Forest precision / recall {extra['precision@0.5']:.2f} / {extra['recall@0.5']:.2f} (threshold 0.5)")
    print(f"  Top-10% scored customers: hit rate {extra['top_decile_hit_rate']:.1%} "
          f"= {extra['top_decile_lift']:.2f}x lift over random")
    print("\n  Feature importance (permutation, ROC-AUC drop):")
    print(importance.sort_values(ascending=False).round(4).to_string())

    pd.DataFrame({"metric": list(auc) + list(extra), "value": list(auc.values()) + list(extra.values())}
                 ).to_csv(out / "model_metrics.csv", index=False)

    # --- scores for every customer (actionable list) ---
    scores = feats[["recency_days", "frequency", "monetary"]].copy()
    scores["segment"] = seg
    scores["repeat_probability"] = rf.predict_proba(feats[MODEL_FEATURES])[:, 1].round(3)
    scores.sort_values("repeat_probability").to_csv(out / "customer_scores.csv")

    make_charts(out, feats, seg, silhouettes, best_k, seg_summary, float(y.mean()), importance)
    print(f"\nSaved tables and charts to: {out.resolve()}")


if __name__ == "__main__":
    main()

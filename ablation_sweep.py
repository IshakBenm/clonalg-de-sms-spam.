#!/usr/bin/env python3
"""
ablation_sweep.py
=======
AIS feature selection + Differential-Evolution-weighted ensemble
for SMS spam detection (Super SMS Dataset, Salman et al. 2024).

Pipeline (inside every training fold, no leakage)
-------------------------------------------------
1. Word (1-2) + char_wb (3-5) TF-IDF, sublinear TF, no stop-word removal
2. Chi2 ranking -> candidate pool of max(3K, K+500) features
3. Clonal-selection AIS wrapper selects exactly K features
   (affinity = spam F1 of Logistic Regression on an inner hold-out)
4. Base classifiers: MultinomialNB, LogisticRegression, calibrated LinearSVC, RandomForest
5. Standard DE (rand/1/bin, F=0.5, CR=0.9) optimizes ensemble weights + threshold
   on out-of-fold probabilities of the training fold

Experiments
-----------
sweep     5x2 cross-validation, K in {50,100,200,500,1000,2000}
          Selectors compared with the same DE ensemble:
            AIS (proposed), Chi2 top-K
          Ensemble ablation on AIS features: DE vs equal-weight average vs single models
          Tests: combined 5x2cv F-test + Wilcoxon, Holm-corrected across K
          Figure: F1 vs K
protocol  Protocol of Salman et al. (2024): stratified 80/20 split, 5 seeds
          TF-IDF+SVM (re-implementation), BoW+QT+LR, all features + DE,
          Chi2+DE and AIS+DE (K = 200 and 8000), side by side with reported values
all       both (default)

Usage
-----
    python ablation_sweep.py --quick                       # fast test (a few minutes)
    python ablation_sweep.py                               # everything
    python ablation_sweep.py --experiment sweep
    python ablation_sweep.py --experiment protocol
    python ablation_sweep.py --experiment sweep --resume   # continue an interrupted sweep
    python ablation_sweep.py --data SMSSpamCollection --experiment sweep   # UCI file also supported
"""

import argparse
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import differential_evolution
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS, CountVectorizer,
                                             TfidfVectorizer)
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]

# Values reported by Salman et al. (2024), Tables 12-13 (single 80/20 split)
REPORTED = [
    ("RoBERTa", 0.974, 0.99, 0.98, 0.98),
    ("TCSVM + Word2Vec", 0.978, 0.99, 0.98, 0.99),
    ("fastText", 0.926, 1.00, 0.92, 0.95),
    ("TCSVM + BoW/TF-IDF", 0.742, 0.99, 0.70, 0.82),
]


# =============================================================================
# 1. DATA
# =============================================================================
TEXT_CANDIDATES = ["SMSes", "sms", "SMS", "text", "Text", "message", "Message", "v2"]
LABEL_CANDIDATES = ["Labels", "labels", "label", "Label", "class", "Class", "v1", "Category"]


def resolve_path(path):
    if os.path.isfile(path):
        return path
    alt = os.path.join(HERE, os.path.basename(path))
    if os.path.isfile(alt):
        return alt
    csvs = [f for f in os.listdir(HERE)
            if f.lower().endswith(".csv") and "result" not in f.lower()]
    if len(csvs) == 1:
        print(f"[info] '{path}' not found, using '{csvs[0]}'")
        return os.path.join(HERE, csvs[0])
    raise FileNotFoundError(f"Dataset not found: {path}\nCSV files in {HERE}: {csvs}\n"
                            f"Use --data with the correct file name.")


def _to_binary(labels):
    s = labels.astype(str).str.strip().str.lower()
    mapping = {"spam": 1, "1": 1, "1.0": 1, "smishing": 1,
               "ham": 0, "0": 0, "0.0": 0, "benign": 0, "legit": 0}
    y = s.map(mapping)
    if y.isna().any():
        raise ValueError(f"Unknown labels: {sorted(s[y.isna()].unique())[:10]}")
    return y.astype(int)


def _norm(t):
    return re.sub(r"\s+", " ", str(t).lower()).strip()


def load_dataset(path, dedup=True, verbose=True):
    """Reads a CSV with text/label columns, or the UCI SMSSpamCollection (label<TAB>text)."""
    path = resolve_path(path)
    df = None
    for enc in ("utf-8", "latin-1"):
        try:
            tmp = pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
        tcol = next((c for c in TEXT_CANDIDATES if c in tmp.columns), None)
        lcol = next((c for c in LABEL_CANDIDATES if c in tmp.columns), None)
        if tcol and lcol:
            df = tmp[[tcol, lcol]].rename(columns={tcol: "text", lcol: "label"})
        break
    if df is None:  # UCI format
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                         encoding="latin-1", quoting=3)

    n0 = len(df)
    df = df.dropna()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    df["label"] = _to_binary(df["label"])

    n_conf = n_dup = 0
    if dedup:
        df["key"] = df["text"].map(_norm)
        nun = df.groupby("key")["label"].nunique()
        conflict = set(nun[nun > 1].index)
        n_conf = len(conflict)
        df = df[~df["key"].isin(conflict)]
        before = len(df)
        df = df.drop_duplicates(subset="key")
        n_dup = before - len(df)

    if verbose:
        print(f"Dataset: {os.path.basename(path)} | raw={n0} | "
              + (f"conflicts removed={n_conf}, duplicates removed={n_dup} | " if dedup
                 else "no deduplication | ")
              + f"final={len(df)} (spam={int(df['label'].sum())}, "
              f"ham={int((df['label'] == 0).sum())}, spam ratio={df['label'].mean():.3f})")
    return np.array(df["text"].tolist(), dtype=object), df["label"].to_numpy()


def paper_preprocess(texts):
    """Preprocessing of Salman et al.: remove special characters and stop words."""
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english"))
    except Exception:
        sw = set(ENGLISH_STOP_WORDS)
    out = []
    for t in texts:
        t = re.sub(r"[^a-z0-9\s]", " ", str(t).lower())
        out.append(" ".join(w for w in t.split() if w not in sw))
    return np.array(out, dtype=object)


# =============================================================================
# 2. FEATURES AND FILTER RANKINGS
# =============================================================================
def vectorize(train_texts, test_texts, max_word=50000, max_char=50000):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95,
                           sublinear_tf=True, lowercase=True,
                           token_pattern=r"(?u)\b\w+\b", max_features=max_word)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, lowercase=True, max_features=max_char)
    Xtr = hstack([word.fit_transform(train_texts), char.fit_transform(train_texts)]).tocsr()
    t0 = time.perf_counter()
    Xte = hstack([word.transform(test_texts), char.transform(test_texts)]).tocsr()
    vec_ms = (time.perf_counter() - t0) * 1000 / max(1, len(test_texts))
    return Xtr, Xte, vec_ms


def chi2_ranking(X, y):
    scores, _ = chi2(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
    order = np.argsort(scores)[::-1]
    return order, scores[order]


def tempered_probs(scores, alpha):
    p = (np.asarray(scores, dtype=float) + 1e-9) ** alpha
    return p / p.sum()


# =============================================================================
# 3. WRAPPER SELECTOR: AIS (proposed)
# =============================================================================
class Affinity:
    """Spam F1 of Logistic Regression on an inner 70/30 hold-out drawn from a
    stratified subsample of the training fold. Cached per subset."""

    def __init__(self, X, y, pool_idx, sample=15000, seed=42):
        n = X.shape[0]
        if n > sample:
            sub, _ = train_test_split(np.arange(n), train_size=sample, stratify=y,
                                      random_state=seed)
        else:
            sub = np.arange(n)
        Xs, ys = X[sub][:, pool_idx], y[sub]
        Xa, Xb, self.ya, self.yb = train_test_split(Xs, ys, test_size=0.3, stratify=ys,
                                                    random_state=seed)
        self.Xa, self.Xb = Xa.tocsc(), Xb.tocsc()
        self.cache = {}

    @property
    def n_evals(self):
        return len(self.cache)

    def f1(self, idx):
        key = idx.tobytes()
        if key not in self.cache:
            clf = LogisticRegression(C=10.0, solver="liblinear", max_iter=1000)
            clf.fit(self.Xa[:, idx], self.ya)
            self.cache[key] = f1_score(self.yb, clf.predict(self.Xb[:, idx]), zero_division=0)
        return self.cache[key]


def _swap_mutate(ab, n_swap, P, probs, rng):
    mask = np.zeros(P, dtype=bool)
    mask[ab] = True
    out = rng.choice(ab, n_swap, replace=False)
    mask[out] = False
    cand = np.setdiff1d(np.flatnonzero(~mask), out, assume_unique=True)
    if len(cand) < n_swap:
        mask[out] = True
        return np.flatnonzero(mask)
    mask[rng.choice(cand, n_swap, replace=False, p=probs[cand] / probs[cand].sum())] = True
    return np.flatnonzero(mask)


class AISFixedK:
    """Clonal selection (CLONALG-style) choosing exactly K features from the pool:
    rank-proportional cloning, affinity-inverse hypermutation (swaps sampled
    with probability ~ chi2^alpha), elitist reselection, receptor editing."""

    def __init__(self, k, pop_size=12, n_gen=15, n_select=4, clone_factor=3, beta=0.15,
                 n_replace=2, alpha=0.5, sample=15000, seed=42):
        self.k, self.pop_size, self.n_gen = k, pop_size, n_gen
        self.n_select, self.clone_factor, self.beta = n_select, clone_factor, beta
        self.n_replace, self.alpha, self.sample, self.seed = n_replace, alpha, sample, seed

    def fit(self, X, y, pool_idx, pool_scores):
        rng = np.random.default_rng(self.seed)
        pool_idx = np.asarray(pool_idx)
        P, k = len(pool_idx), min(self.k, len(pool_idx))
        self.history_, self.n_evals_ = [], 0
        if P <= k:
            self.selected_ = pool_idx[:k]
            return self
        aff = Affinity(X, y, pool_idx, self.sample, self.seed)
        probs = tempered_probs(pool_scores, self.alpha)

        def new():
            return np.sort(rng.choice(P, k, replace=False, p=probs))

        pop = [np.arange(k)] + [new() for _ in range(self.pop_size - 1)]
        fit = [aff.f1(a) for a in pop]
        for g in range(self.n_gen):
            order = np.argsort(fit)[::-1]
            pop, fit = [pop[i] for i in order], [fit[i] for i in order]
            fmin, fmax = min(fit), max(fit)
            clones, cfit = [], []
            for rank in range(min(self.n_select, len(pop))):
                n_cl = max(1, int(round(self.clone_factor * self.n_select / (rank + 1))))
                rate = self.beta * np.exp(-2.0 * (fit[rank] - fmin) / (fmax - fmin + 1e-12))
                n_swap = min(max(1, int(rate * k)), k, P - k)
                for _ in range(n_cl):
                    c = _swap_mutate(pop[rank], n_swap, P, probs, rng)
                    clones.append(c)
                    cfit.append(aff.f1(c))
            allp, allf = pop + clones, fit + cfit
            order = np.argsort(allf)[::-1][: self.pop_size]
            pop, fit = [allp[i] for i in order], [allf[i] for i in order]
            for j in range(1, self.n_replace + 1):
                pop[-j] = new()
                fit[-j] = aff.f1(pop[-j])
            self.history_.append(float(max(fit)))
        best = pop[int(np.argmax(fit))]
        self.selected_ = pool_idx[best]
        self.n_evals_ = aff.n_evals
        return self


# =============================================================================
# 4. DE-WEIGHTED ENSEMBLE
# =============================================================================
BASE_NAMES = ["NB", "LR", "SVM", "RF"]


def make_base_models(seed, rf_trees=200):
    return {
        "NB": MultinomialNB(alpha=0.1),
        "LR": LogisticRegression(C=10.0, solver="liblinear", max_iter=2000, random_state=seed),
        "SVM": CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed), cv=3, method="sigmoid"),
        "RF": RandomForestClassifier(n_estimators=rf_trees, n_jobs=-1, random_state=seed),
    }


def de_optimize_weights(P_oof, y, args, seed):
    m = P_oof.shape[1]
    bounds = [(0.0, 1.0)] * m + [(0.2, 0.8)]

    def objective(v):
        w = v[:m]
        s = w.sum()
        if s <= 1e-9:
            return 1.0
        return -f1_score(y, ((P_oof @ (w / s)) >= v[m]).astype(int), zero_division=0)

    res = differential_evolution(objective, bounds, strategy="rand1bin", mutation=0.5,
                                 recombination=0.9, popsize=args.de_pop, maxiter=args.de_iter,
                                 tol=1e-6, polish=False, seed=seed)
    w = res.x[:m]
    w = w / w.sum() if w.sum() > 0 else np.full(m, 1.0 / m)
    return w, float(res.x[m])


def run_ensemble(Xtr, ytr, Xte, args, seed, full=False):
    """Returns {name: (test_scores, threshold)}, DE weights, threshold, train s, infer ms/msg."""
    t0 = time.perf_counter()
    models = make_base_models(seed, args.rf_trees)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    P_oof = np.column_stack([
        cross_val_predict(clone(models[n]), Xtr, ytr, cv=inner, method="predict_proba")[:, 1]
        for n in BASE_NAMES])
    fitted = {n: clone(models[n]).fit(Xtr, ytr) for n in BASE_NAMES}
    w, thr = de_optimize_weights(P_oof, ytr, args, seed)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    P_te = np.column_stack([fitted[n].predict_proba(Xte)[:, 1] for n in BASE_NAMES])
    infer_ms = (time.perf_counter() - t1) * 1000 / max(1, Xte.shape[0])
    out = {"DE": (P_te @ w, thr)}
    if full:
        out["AVG"] = (P_te.mean(axis=1), 0.5)
        for i, n in enumerate(BASE_NAMES):
            out[n] = (P_te[:, i], 0.5)
    return out, dict(zip(BASE_NAMES, np.round(w, 4).tolist())), thr, train_s, infer_ms


# =============================================================================
# 5. BASELINES
# =============================================================================
def senol_baseline(t_tr, y_tr, t_te, seed):
    t0 = time.perf_counter()
    bow = CountVectorizer()
    qt = QuantileTransformer(n_quantiles=1000, random_state=seed)
    lr = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    lr.fit(qt.fit_transform(bow.fit_transform(t_tr)), y_tr)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    prob = lr.predict_proba(qt.transform(bow.transform(t_te)))[:, 1]
    return prob, 0.5, train_s, (time.perf_counter() - t1) * 1000 / max(1, len(t_te))


def tfidf_svm_baseline(t_tr, y_tr, t_te, seed):
    t0 = time.perf_counter()
    vec = TfidfVectorizer()
    svm = LinearSVC(C=1.0, random_state=seed).fit(vec.fit_transform(t_tr), y_tr)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    score = svm.decision_function(vec.transform(t_te))
    return score, 0.0, train_s, (time.perf_counter() - t1) * 1000 / max(1, len(t_te))


# =============================================================================
# 6. METRICS AND STATISTICS
# =============================================================================
def evaluate(y_true, score, thr=0.5):
    pred = (np.asarray(score) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    try:
        auc = roc_auc_score(y_true, score)
    except ValueError:
        auc = np.nan
    hp = tn / (tn + fn) if tn + fn else 0.0
    hr = tn / (tn + fp) if tn + fp else 0.0
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, pred), "auc": auc,
            "ham_f1": 2 * hp * hr / (hp + hr) if hp + hr else 0.0,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def five_by_two_f_test(d):
    """Combined 5x2cv F-test (Alpaydin, 1999); d has shape (5, 2)."""
    d = np.asarray(d, dtype=float)
    s2 = ((d - d.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    denom = 2.0 * s2.sum()
    if denom <= 0:
        return np.nan, np.nan
    F = (d ** 2).sum() / denom
    return float(F), float(1.0 - stats.f.cdf(F, 10, 5))


def wilcoxon_p(a, b):
    try:
        return float(stats.wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def holm(pvals):
    p = np.nan_to_num(np.asarray(pvals, dtype=float), nan=1.0)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


# =============================================================================
# 7. EXPERIMENT A: SWEEP (5x2 CV)
# =============================================================================
def run_sweep(args):
    out = os.path.join(args.out, "sweep")
    os.makedirs(out, exist_ok=True)
    print("\n" + "#" * 76 + "\n# EXPERIMENT A: SELECTOR COMPARISON ACROSS FEATURE BUDGETS\n" + "#" * 76)
    texts, y = load_dataset(args.data, dedup=True)
    if args.quick and len(y) > 8000:
        keep, _ = train_test_split(np.arange(len(y)), train_size=8000, stratify=y,
                                   random_state=args.seed)
        texts, y = texts[keep], y[keep]

    zeros = np.zeros(len(y))
    if args.cv == "5x2":
        splits = [(rep, f, tr, te) for rep in range(5)
                  for f, (tr, te) in enumerate(StratifiedKFold(
                      2, shuffle=True, random_state=args.seed + rep).split(zeros, y))]
    else:
        splits = [(0, f, tr, te) for f, (tr, te) in enumerate(StratifiedKFold(
            args.folds, shuffle=True, random_state=args.seed).split(zeros, y))]

    csv_path = os.path.join(out, "results_per_run.csv")
    done_path = os.path.join(out, "done_runs.txt")
    rows, done = [], set()
    if args.resume and os.path.isfile(csv_path) and os.path.isfile(done_path):
        done = {int(x) for x in open(done_path).read().split()}
        rows = pd.read_csv(csv_path).query("run in @done").to_dict("records")
        print(f"[resume] {len(done)} completed runs loaded")
    elif os.path.isfile(done_path):
        os.remove(done_path)

    wlog, t_start = [], time.time()
    for run, (rep, fold, tr, te) in enumerate(splits, 1):
        if run in done:
            continue
        seed = args.seed + run
        print(f"\n=== RUN {run}/{len(splits)} (repetition {rep + 1}, fold {fold + 1}) "
              f"train={len(tr)} test={len(te)} | elapsed {time.time() - t_start:.0f}s")
        t_tr, t_te, y_tr, y_te = texts[tr], texts[te], y[tr], y[te]
        base = {"run": run, "rep": rep, "fold": fold}

        prob, thr, a, b = senol_baseline(t_tr, y_tr, t_te, seed)
        rows.append({**base, "K": "all", "method": "BoW+QT+LR (baseline)",
                     "n_features": np.nan, **evaluate(y_te, prob, thr),
                     "train_s": a, "infer_ms": b})

        Xtr, Xte, vec_ms = vectorize(t_tr, t_te, args.max_word, args.max_char)
        c_idx, c_sc = chi2_ranking(Xtr, y_tr)
        nf = Xtr.shape[1]
        print(f"  features: {nf}")

        def add(K, method, sel, extra_s=0.0, full=False, extra=None):
            sel = np.sort(sel)
            res, w, thr, tr_s, inf_ms = run_ensemble(Xtr[:, sel], y_tr, Xte[:, sel], args,
                                                     seed, full=full)
            for key, (sc, t) in res.items():
                name = method if key == "DE" else method.replace("+DE", f"+{key}")
                rows.append({**base, "K": str(K), "method": name, "n_features": len(sel),
                             **evaluate(y_te, sc, t),
                             "train_s": tr_s + extra_s if key == "DE" else np.nan,
                             "infer_ms": inf_ms + vec_ms if key == "DE" else np.nan,
                             **(extra or {})})
            wlog.append({**base, "K": K, "method": method, "weights": w, "threshold": round(thr, 4)})
            r = next(x for x in reversed(rows) if x["method"] == method and x["run"] == run)
            print(f"    {method:<10} F1={r['f1']:.4f}  MCC={r['mcc']:.4f}")

        for K in args.k:
            K = min(K, nf)
            P = min(nf, max(args.pool_factor * K, K + args.pool_extra))
            print(f"  --- K={K} (chi2 pool {P})")
            t0 = time.time()
            ais = AISFixedK(K, pop_size=args.ais_pop, n_gen=args.ais_gen, alpha=args.alpha,
                            sample=args.ais_sample, seed=seed).fit(Xtr, y_tr, c_idx[:P], c_sc[:P])
            t_ais = time.time() - t0
            ov = len(np.intersect1d(ais.selected_, c_idx[:K])) / K
            add(K, "AIS+DE", ais.selected_, t_ais, full=True,
                extra={"search_evals": ais.n_evals_, "chi2_overlap": ov})

            add(K, "Chi2+DE", c_idx[:K])

        pd.DataFrame(rows).to_csv(csv_path, index=False)
        with open(done_path, "a") as f:
            f.write(f"{run}\n")

    df = pd.DataFrame(rows)
    df["K"] = df["K"].astype(str)
    df.to_csv(csv_path, index=False)
    with open(os.path.join(out, "de_weights.json"), "w") as f:
        json.dump(wlog, f, indent=1, default=str)

    # ---- summary
    order_k = ["all"] + [str(min(k, 10 ** 9)) for k in args.k]
    summ = df.groupby(["K", "method"], sort=False)[METRICS + ["train_s", "infer_ms"]].agg(["mean", "std"])
    summ = summ.reindex([i for k in order_k for i in summ.index if i[0] == k])
    summ.to_csv(os.path.join(out, "summary_raw.csv"))
    pretty = pd.DataFrame(index=summ.index)
    for m in ["accuracy", "f1", "mcc", "auc"]:
        pretty[m] = [f"{a:.4f} ± {b:.4f}" for a, b in
                     zip(summ[(m, "mean")], summ[(m, "std")].fillna(0))]
    pretty["train_s"] = summ[("train_s", "mean")].round(1)
    pretty.to_csv(os.path.join(out, "summary_table.csv"))

    # ---- statistical tests (F1), Holm-corrected within each family
    def compare(K, ma, mb, kb=None):
        a = df[(df.K == K) & (df.method == ma)].set_index("run")["f1"]
        b = df[(df.K == (kb or K)) & (df.method == mb)].set_index("run")["f1"]
        common = a.index.intersection(b.index).sort_values()
        a, b = a[common].values, b[common].values
        d = a - b
        F, pF = (five_by_two_f_test(d.reshape(5, 2)) if args.cv == "5x2" and len(d) == 10
                 else (np.nan, np.nan))
        return {"K": K, "A": ma, "B": mb, "mean_A": a.mean(), "mean_B": b.mean(),
                "diff": d.mean(), "A_wins": int((d > 0).sum()), "n": len(d),
                "p_wilcoxon": wilcoxon_p(a, b), "p_5x2cv_F": pF}

    tests = []
    ks = [str(k) for k in args.k]
    families = [("AIS+DE", "Chi2+DE"), ("AIS+DE", "AIS+AVG")]
    for ma, mb in families:
        fam = [compare(K, ma, mb) for K in ks if ((df.K == K) & (df.method == mb)).any()]
        for key in ("p_wilcoxon", "p_5x2cv_F"):
            adj = holm([r[key] for r in fam])
            for r, v in zip(fam, adj):
                r[key + "_holm"] = v
        tests += fam
    # DE vs best single model (chosen by mean F1 at each K)
    fam = []
    for K in ks:
        singles = {f"AIS+{n}": df[(df.K == K) & (df.method == f"AIS+{n}")]["f1"].mean()
                   for n in BASE_NAMES}
        fam.append(compare(K, "AIS+DE", max(singles, key=singles.get)))
    for key in ("p_wilcoxon", "p_5x2cv_F"):
        for r, v in zip(fam, holm([r[key] for r in fam])):
            r[key + "_holm"] = v
    tests += fam
    r = compare(ks[-1], "AIS+DE", "BoW+QT+LR (baseline)", kb="all")
    r["p_wilcoxon_holm"], r["p_5x2cv_F_holm"] = r["p_wilcoxon"], r["p_5x2cv_F"]
    tests.append(r)
    tests = pd.DataFrame(tests)
    tests.to_csv(os.path.join(out, "statistical_tests.csv"), index=False)

    print("\n" + "=" * 76 + f"\nEXPERIMENT A SUMMARY ({df.run.nunique()} runs)\n" + "=" * 76)
    with pd.option_context("display.max_rows", None, "display.width", 220,
                           "display.max_columns", None):
        print(pretty)
        cols = ["K", "A", "B", "diff", "A_wins", "p_wilcoxon_holm", "p_5x2cv_F_holm"]
        print("\nTESTS on F1 (Holm-corrected across K within each comparison)")
        print(tests[[c for c in cols if c in tests]].round(5).to_string(index=False))

    # ---- figure: F1 vs K
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({"font.family": "serif", "font.size": 10})
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        styles = {"AIS+DE": ("o-", "tab:red", "CLONALG + DE (proposed)"),
                  "Chi2+DE": ("s--", "tab:blue", r"$\chi^2$ top-K + DE")}
        kk = [int(k) for k in ks]
        for m, (fmt, col, lab) in styles.items():
            sub = df[df.method == m]
            if sub.empty:
                continue
            g = sub.groupby("K")["f1"].agg(["mean", "std"]).reindex(ks)
            ax.errorbar(kk, g["mean"], yerr=g["std"], fmt=fmt, color=col, capsize=3,
                        ms=5, lw=1.4, label=lab)
        ax.axhline(df[df.K == "all"]["f1"].mean(), color="black", ls=":", lw=1,
                   label="BoW+QT+LR (all features)")
        sig = tests[(tests.A == "AIS+DE") & (tests.B == "Chi2+DE")]
        for _, r in sig.iterrows():
            ok = r["p_wilcoxon_holm"] < 0.05 and (np.isnan(r["p_5x2cv_F_holm"])
                                                  or r["p_5x2cv_F_holm"] < 0.05)
            if ok:
                ax.annotate("*", (int(r.K), r.mean_A), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=14, color="tab:red")
        ax.set_xscale("log")
        ax.set_xticks(kk)
        ax.set_xticklabels(ks)
        ax.set_xlabel("Number of selected features K")
        ax.set_ylabel("F1-score (spam)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"f1_vs_k.{ext}"), dpi=300)
        print(f"\nFigure saved: {os.path.join(out, 'f1_vs_k.png')}  (* = AIS > chi2, "
              f"Holm-corrected p < 0.05 on both tests)")
    except ImportError:
        print("matplotlib not installed: figure skipped")


# =============================================================================
# 8. EXPERIMENT B: PROTOCOL OF SALMAN et al. (2024)
# =============================================================================
def run_protocol(args):
    out = os.path.join(args.out, "protocol")
    os.makedirs(out, exist_ok=True)
    print("\n" + "#" * 76 + "\n# EXPERIMENT B: PROTOCOL OF SALMAN et al. (2024)\n" + "#" * 76)
    texts, y = load_dataset(args.data, dedup=False)
    if args.quick and len(y) > 8000:
        keep, _ = train_test_split(np.arange(len(y)), train_size=8000, stratify=y,
                                   random_state=args.seed)
        texts, y = texts[keep], y[keep]
    variants = {"raw": texts}
    if args.protocol_prep in ("paper", "both"):
        variants["paper"] = paper_preprocess(texts)
        if args.protocol_prep == "paper":
            del variants["raw"]

    rows, t_start = [], time.time()
    csv_path = os.path.join(out, "results_per_seed.csv")
    for seed in args.seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        y_tr, y_te = y[tr], y[te]
        for prep, T in variants.items():
            t_tr, t_te = T[tr], T[te]
            print(f"\n=== seed {seed} | preprocessing={prep} | train={len(tr)} test={len(te)} "
                  f"| elapsed {time.time() - t_start:.0f}s")
            base = {"seed": seed, "preprocessing": prep}

            def add(method, nfeat, sc, thr, a, b):
                m = evaluate(y_te, sc, thr)
                rows.append({**base, "method": method, "n_features": nfeat, **m,
                             "train_s": a, "infer_ms": b})
                print(f"  {method:<26} n={str(nfeat):<7} ACC={m['accuracy']:.4f} "
                      f"F1={m['f1']:.4f} MCC={m['mcc']:.4f}")

            add("TF-IDF + SVM (re-impl.)", "all", *tfidf_svm_baseline(t_tr, y_tr, t_te, seed))
            add("BoW+QT+LR", "all", *bow_wrap(t_tr, y_tr, t_te, seed))

            Xtr, Xte, vec_ms = vectorize(t_tr, t_te, args.max_word, args.max_char)
            c_idx, c_sc = chi2_ranking(Xtr, y_tr)
            nf = Xtr.shape[1]

            def add_ens(method, sel, extra_s=0.0):
                sel = np.sort(sel)
                res, _, _, tr_s, inf_ms = run_ensemble(Xtr[:, sel], y_tr, Xte[:, sel], args, seed)
                add(method, len(sel), *res["DE"], tr_s + extra_s, inf_ms + vec_ms)

            if args.full_features:
                add_ens("All features + DE", np.arange(nf))
            for K in args.protocol_k:
                K = min(K, nf)
                P = min(nf, max(args.pool_factor * K, K + args.pool_extra))
                add_ens(f"Chi2+DE (K={K})", c_idx[:K])
                t0 = time.time()
                ais = AISFixedK(K, pop_size=args.ais_pop, n_gen=args.ais_gen, alpha=args.alpha,
                                sample=args.ais_sample, seed=seed).fit(Xtr, y_tr, c_idx[:P], c_sc[:P])
                add_ens(f"AIS+DE (K={K})", ais.selected_, time.time() - t0)
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    cols = ["accuracy", "precision", "recall", "f1", "mcc", "ham_f1", "train_s", "infer_ms"]
    g = df.groupby(["preprocessing", "method"], sort=False)
    mean, std = g[cols].mean(), g[cols].std()
    table = []
    for name, acc, p, r, f in REPORTED:
        table.append({"source": "Salman et al. 2024 (reported)", "method": name,
                      "accuracy": f"{100 * acc:.1f}", "precision": f"{100 * p:.0f}",
                      "recall": f"{100 * r:.0f}", "f1": f"{100 * f:.0f}"})
    for idx in mean.index:
        row = {"source": f"this work ({idx[0]}, {len(args.seeds)} seeds)", "method": idx[1],
               "n_features": g["n_features"].first()[idx]}
        for c in ["accuracy", "precision", "recall", "f1", "mcc", "ham_f1"]:
            sd = std.loc[idx, c]
            row[c] = f"{100 * mean.loc[idx, c]:.2f}" + ("" if pd.isna(sd) else f" ± {100 * sd:.2f}")
        row["train_s"] = f"{mean.loc[idx, 'train_s']:.1f}"
        row["infer_ms"] = f"{mean.loc[idx, 'infer_ms']:.4f}"
        table.append(row)
    table = pd.DataFrame(table).fillna("")
    table.to_csv(os.path.join(out, "comparison_with_salman.csv"), index=False)

    print("\n" + "=" * 76 + "\nEXPERIMENT B: COMPARISON WITH SALMAN et al. (2024), values in %\n" + "=" * 76)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(table.to_string(index=False))
    print("\nprecision/recall/F1/MCC of 'this work': spam = positive class; ham_f1: ham = positive.\n"
          "Accuracy and MCC are the most directly comparable metrics.")


def bow_wrap(t_tr, y_tr, t_te, seed):
    return senol_baseline(t_tr, y_tr, t_te, seed)


# =============================================================================
# 9. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="AIS + DE-weighted ensemble for SMS spam detection")
    ap.add_argument("--data", default="super_sms_dataset.csv")
    ap.add_argument("--out", default="results_final")
    ap.add_argument("--experiment", choices=["sweep", "protocol", "all"], default="all")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-word", type=int, default=50000)
    ap.add_argument("--max-char", type=int, default=50000)
    # sweep
    ap.add_argument("--cv", choices=["5x2", "kfold"], default="5x2")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--k", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000])
    ap.add_argument("--resume", action="store_true")
    # protocol
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--protocol-k", type=int, nargs="+", default=[200, 8000])
    ap.add_argument("--protocol-prep", choices=["raw", "paper", "both"], default="raw")
    ap.add_argument("--no-full", dest="full_features", action="store_false")
    # AIS
    ap.add_argument("--pool-factor", type=int, default=3)
    ap.add_argument("--pool-extra", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--ais-pop", type=int, default=12)
    ap.add_argument("--ais-gen", type=int, default=15)
    ap.add_argument("--ais-sample", type=int, default=15000)
    # ensemble
    ap.add_argument("--de-pop", type=int, default=15)
    ap.add_argument("--de-iter", type=int, default=50)
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="fast test on a subsample")
    ap.add_argument("--no-show", dest="show", action="store_false")
    args = ap.parse_args()

    if args.quick:
        args.out = args.out + "_quick"
        args.cv, args.folds, args.k = "kfold", 2, [50, 200]
        args.seeds, args.protocol_k = [42], [200]
        args.ais_gen, args.ais_pop, args.de_iter, args.rf_trees = 4, 8, 15, 50
        args.max_word = args.max_char = 20000

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    t0 = time.time()
    if args.experiment in ("sweep", "all"):
        run_sweep(args)
    if args.experiment in ("protocol", "all"):
        run_protocol(args)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min | results in {os.path.abspath(args.out)}")

    if args.show:
        try:
            import matplotlib.pyplot as plt
            if plt.get_fignums():
                plt.show()
        except ImportError:
            pass


if __name__ == "__main__":
    main()
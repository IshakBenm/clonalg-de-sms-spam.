#!/usr/bin/env python3
"""
main_clonalg_de.py
==================
Proposed approach ONLY: CLONALG feature selection + Differential-Evolution-weighted ensemble
for SMS spam detection.

Pipeline (fitted on the training data only)
-------------------------------------------
1. TF-IDF: word 1-2 grams + char_wb 3-5 grams, sublinear TF, no stop-word removal
2. Chi-square candidate pool of max(3K, K+500) features
3. CLONALG wrapper selects exactly K features
   (affinity = spam F1 of Logistic Regression on an inner hold-out)
4. Base classifiers: MultinomialNB, LogisticRegression, calibrated LinearSVC, RandomForest
5. DE (rand/1/bin, F=0.5, CR=0.9) optimizes the 4 weights + decision threshold
   on out-of-fold probabilities
K = "all" skips steps 2-3 (all features + DE ensemble).

Experiments
-----------
budget   Super SMS, 5x2 cross-validation, K in {50,100,200,500,1000,2000}
salman   Super SMS, protocol of Salman et al. (2024): stratified 80/20 split, 5 seeds,
         K in {200, 8000, all}; printed next to the results they report
senol    UCI SMS Spam Collection, protocol of Senol et al. (2026): stratified 70/30 split,
         seed 111 (their split) + 4 seeds, K in {100, 2000, all}; printed next to their results;
         also saves the words selected by CLONALG

Usage
-----
    python main_clonalg_de.py --quick                                   # fast test
    python main_clonalg_de.py --experiment budget  --data super_sms_dataset.csv
    python main_clonalg_de.py --experiment salman  --data super_sms_dataset.csv
    python main_clonalg_de.py --experiment senol   --uci SMSSpamCollection
    python main_clonalg_de.py --experiment all     --data super_sms_dataset.csv --uci SMSSpamCollection
"""

import argparse
import json
import os
import re
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.sparse import hstack
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]

# Results reported in the literature (not rerun here)
SALMAN_REPORTED = [  # Salman et al. 2024, Super SMS, 80/20 split (accuracy)
    ("RoBERTa", 0.974), ("TCSVM + Word2Vec", 0.978), ("fastText", 0.926),
    ("TCSVM + BoW/TF-IDF", 0.742)]
SENOL_REPORTED = [  # Senol et al. 2026, UCI, 70/30 stratified split, seed 111
    # (method, n_features, accuracy, precision, recall, f1, auc)
    ("BoW + QuantileTransformer + LR (best overall)", "all", 0.9886, 0.9952, 0.9196, 0.9559, 0.9905),
    ("BoW + LASSO + XGB (best feature selection)", "144", 0.9820, 0.9663, 0.8973, 0.9306, 0.9835),
    ("BoW + Chi2 + XGB", "100", 0.9814, 0.9573, 0.9018, 0.9287, 0.9848),
    ("TF-IDF + MI + GB", "100", 0.9838, 0.9758, 0.9018, 0.9374, 0.9826),
    ("BoW + XGB (best ensemble)", "all", 0.9838, 0.9758, 0.9018, 0.9374, 0.9839)]


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
    raise FileNotFoundError(f"Dataset not found: {path} (also looked in {HERE})")


def _to_binary(labels):
    s = labels.astype(str).str.strip().str.lower()
    y = s.map({"spam": 1, "1": 1, "1.0": 1, "smishing": 1,
               "ham": 0, "0": 0, "0.0": 0, "benign": 0, "legit": 0})
    if y.isna().any():
        raise ValueError(f"Unknown labels: {sorted(s[y.isna()].unique())[:10]}")
    return y.astype(int)


def load_dataset(path, dedup=True):
    """CSV with text/label columns, or UCI SMSSpamCollection (label<TAB>text)."""
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
    if df is None:
        df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"],
                         encoding="latin-1", quoting=3)
    n0 = len(df)
    df = df.dropna()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    df["label"] = _to_binary(df["label"])
    if dedup:
        df["key"] = df["text"].map(lambda t: re.sub(r"\s+", " ", t.lower()).strip())
        nun = df.groupby("key")["label"].nunique()
        df = df[~df["key"].isin(set(nun[nun > 1].index))].drop_duplicates(subset="key")
    print(f"Dataset: {os.path.basename(path)} | raw={n0} | "
          f"{'deduplicated' if dedup else 'no deduplication'} | final={len(df)} "
          f"(spam={int(df['label'].sum())}, spam ratio={df['label'].mean():.3f})")
    return np.array(df["text"].tolist(), dtype=object), df["label"].to_numpy()


def senol_preprocess(texts):
    """Preprocessing of Senol et al.: lowercase, URL->url, digits->num,
    remove non-alphabetic characters, normalize whitespace."""
    out = []
    for t in texts:
        t = re.sub(r"(https?://\S+|www\.\S+)", " url ", str(t).lower())
        t = re.sub(r"[^a-z\s]", " ", re.sub(r"\d+", " num ", t))
        out.append(re.sub(r"\s+", " ", t).strip())
    return np.array(out, dtype=object)


# =============================================================================
# 2. FEATURES
# =============================================================================
def vectorize(train_texts, test_texts, max_word=50000, max_char=50000):
    word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95,
                           sublinear_tf=True, token_pattern=r"(?u)\b\w+\b",
                           max_features=max_word)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, max_features=max_char)
    Xtr = hstack([word.fit_transform(train_texts), char.fit_transform(train_texts)]).tocsr()
    t0 = time.perf_counter()
    Xte = hstack([word.transform(test_texts), char.transform(test_texts)]).tocsr()
    vec_ms = (time.perf_counter() - t0) * 1000 / max(1, len(test_texts))
    names = np.array([f"w:{n}" for n in word.get_feature_names_out()] +
                     [f"c:{n}" for n in char.get_feature_names_out()], dtype=object)
    return Xtr, Xte, vec_ms, names


def chi2_pool(X, y, size):
    scores, _ = chi2(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
    order = np.argsort(scores)[::-1][:size]
    return order, scores[order]


# =============================================================================
# 3. CLONALG FEATURE SELECTION
# =============================================================================
class Affinity:
    """Spam F1 of Logistic Regression on an inner 70/30 hold-out (stratified subsample)."""

    def __init__(self, X, y, pool_idx, sample=15000, seed=42):
        n = X.shape[0]
        sub = (train_test_split(np.arange(n), train_size=sample, stratify=y,
                                random_state=seed)[0] if n > sample else np.arange(n))
        Xs, ys = X[sub][:, pool_idx], y[sub]
        Xa, Xb, self.ya, self.yb = train_test_split(Xs, ys, test_size=0.3, stratify=ys,
                                                    random_state=seed)
        self.Xa, self.Xb = Xa.tocsc(), Xb.tocsc()
        self.cache = {}

    def f1(self, idx):
        key = idx.tobytes()
        if key not in self.cache:
            clf = LogisticRegression(C=10.0, solver="liblinear", max_iter=1000)
            clf.fit(self.Xa[:, idx], self.ya)
            self.cache[key] = f1_score(self.yb, clf.predict(self.Xb[:, idx]), zero_division=0)
        return self.cache[key]


class CLONALGSelector:
    """CLONALG (de Castro & Von Zuben, 2002) adapted to fixed-size feature subset selection:
    antibody = K features from the chi2 pool; rank-proportional cloning; hypermutation by
    feature swaps (sampling ~ chi2^alpha) at a rate inversely proportional to affinity;
    elitist reselection; receptor editing of the d worst antibodies."""

    def __init__(self, k, pop_size=12, n_gen=15, n_select=4, clone_factor=3, beta=0.15,
                 n_replace=2, alpha=0.5, sample=15000, seed=42):
        self.k, self.pop_size, self.n_gen = k, pop_size, n_gen
        self.n_select, self.clone_factor, self.beta = n_select, clone_factor, beta
        self.n_replace, self.alpha, self.sample, self.seed = n_replace, alpha, sample, seed

    def fit(self, X, y, pool_idx, pool_scores):
        rng = np.random.default_rng(self.seed)
        pool_idx = np.asarray(pool_idx)
        P, k = len(pool_idx), min(self.k, len(pool_idx))
        self.history_ = []
        if P <= k:
            self.selected_ = pool_idx[:k]
            return self
        aff = Affinity(X, y, pool_idx, self.sample, self.seed)
        p = (np.asarray(pool_scores, float) + 1e-9) ** self.alpha
        probs = p / p.sum()

        def new():
            return np.sort(rng.choice(P, k, replace=False, p=probs))

        def mutate(ab, n_swap):
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

        pop = [np.arange(k)] + [new() for _ in range(self.pop_size - 1)]
        fit = [aff.f1(a) for a in pop]
        for _ in range(self.n_gen):
            order = np.argsort(fit)[::-1]
            pop, fit = [pop[i] for i in order], [fit[i] for i in order]
            fmin, fmax = min(fit), max(fit)
            clones, cfit = [], []
            for rank in range(min(self.n_select, len(pop))):
                n_cl = max(1, int(round(self.clone_factor * self.n_select / (rank + 1))))
                rate = self.beta * np.exp(-2.0 * (fit[rank] - fmin) / (fmax - fmin + 1e-12))
                n_swap = min(max(1, int(rate * k)), k, P - k)
                for _ in range(n_cl):
                    c = mutate(pop[rank], n_swap)
                    clones.append(c)
                    cfit.append(aff.f1(c))
            allp, allf = pop + clones, fit + cfit
            order = np.argsort(allf)[::-1][: self.pop_size]
            pop, fit = [allp[i] for i in order], [allf[i] for i in order]
            for j in range(1, self.n_replace + 1):
                pop[-j] = new()
                fit[-j] = aff.f1(pop[-j])
            self.history_.append(float(max(fit)))
        self.selected_ = pool_idx[pop[int(np.argmax(fit))]]
        self.n_evals_ = len(aff.cache)
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


def de_weights(P_oof, y, args, seed):
    m = P_oof.shape[1]

    def objective(v):
        w = v[:m]
        s = w.sum()
        if s <= 1e-9:
            return 1.0
        return -f1_score(y, ((P_oof @ (w / s)) >= v[m]).astype(int), zero_division=0)

    res = differential_evolution(objective, [(0.0, 1.0)] * m + [(0.2, 0.8)],
                                 strategy="rand1bin", mutation=0.5, recombination=0.9,
                                 popsize=args.de_pop, maxiter=args.de_iter, tol=1e-6,
                                 polish=False, seed=seed)
    w = res.x[:m]
    w = w / w.sum() if w.sum() > 0 else np.full(m, 1.0 / m)
    return w, float(res.x[m])


def de_ensemble(Xtr, ytr, Xte, args, seed):
    t0 = time.perf_counter()
    models = make_base_models(seed, args.rf_trees)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    P_oof = np.column_stack([
        cross_val_predict(clone(models[n]), Xtr, ytr, cv=inner, method="predict_proba")[:, 1]
        for n in BASE_NAMES])
    fitted = {n: clone(models[n]).fit(Xtr, ytr) for n in BASE_NAMES}
    w, thr = de_weights(P_oof, ytr, args, seed)
    train_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    score = np.column_stack([fitted[n].predict_proba(Xte)[:, 1] for n in BASE_NAMES]) @ w
    infer_ms = (time.perf_counter() - t1) * 1000 / max(1, Xte.shape[0])
    return score, thr, dict(zip(BASE_NAMES, np.round(w, 4).tolist())), train_s, infer_ms


# =============================================================================
# 5. ONE RUN OF THE PROPOSED APPROACH
# =============================================================================
def evaluate(y_true, score, thr):
    pred = (np.asarray(score) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, pred),
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "mcc": matthews_corrcoef(y_true, pred), "auc": roc_auc_score(y_true, score),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def run_approach(t_tr, y_tr, t_te, y_te, ks, args, seed, base, rows, feats=None):
    """Runs CLONALG+DE for every K in ks ('all' = all features + DE) on one split."""
    Xtr, Xte, vec_ms, names = vectorize(t_tr, t_te, args.max_word, args.max_char)
    nf = Xtr.shape[1]
    print(f"  TF-IDF features: {nf}")
    for K in ks:
        t0 = time.time()
        if str(K) == "all":
            sel, label, n_evals = np.arange(nf), "all features + DE", 0
        else:
            K = min(int(K), nf)
            P = min(nf, max(args.pool_factor * K, K + args.pool_extra))
            pool, sc = chi2_pool(Xtr, y_tr, P)
            clg = CLONALGSelector(K, pop_size=args.pop, n_gen=args.gen, alpha=args.alpha,
                                  sample=args.sample, seed=seed).fit(Xtr, y_tr, pool, sc)
            sel, label = np.sort(clg.selected_), f"CLONALG+DE (K={K})"
            n_evals = getattr(clg, "n_evals_", 0)
            if feats is not None and K <= 200:
                feats[f"{base}_K{K}".replace(" ", "")] = sorted(names[sel].tolist())
        t_sel = time.time() - t0
        score, thr, w, tr_s, inf_ms = de_ensemble(Xtr[:, sel], y_tr, Xte[:, sel], args, seed)
        m = evaluate(y_te, score, thr)
        rows.append({**base, "K": str(K), "method": label, "n_features": len(sel), **m,
                     "train_s": t_sel + tr_s, "infer_ms": inf_ms + vec_ms,
                     "weights": json.dumps(w), "threshold": round(thr, 4),
                     "affinity_evals": n_evals})
        print(f"  {label:<24} ACC={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} MCC={m['mcc']:.4f}  ({t_sel + tr_s:.0f}s)")


def summarize(df, group, out_csv):
    g = df.groupby(group, sort=False)
    mean, std = g[METRICS + ["train_s", "infer_ms"]].mean(), g[METRICS].std()
    t = pd.DataFrame(index=mean.index)
    t["n_features"] = g["n_features"].first()
    for m in METRICS:
        t[m] = [f"{a:.4f} ± {b:.4f}" if not pd.isna(b) else f"{a:.4f}"
                for a, b in zip(mean[m], std[m])]
    t["train_s"], t["infer_ms"] = mean["train_s"].round(1), mean["infer_ms"].round(4)
    t.to_csv(out_csv)
    return t


# =============================================================================
# 6. EXPERIMENTS
# =============================================================================
def exp_budget(args):
    out = os.path.join(args.out, "budget_super")
    os.makedirs(out, exist_ok=True)
    print("\n" + "#" * 70 + "\n# BUDGET ANALYSIS - Super SMS, 5x2 CV\n" + "#" * 70)
    texts, y = load_dataset(args.data, dedup=True)
    if args.quick and len(y) > 8000:
        keep = train_test_split(np.arange(len(y)), train_size=8000, stratify=y,
                                random_state=args.seed)[0]
        texts, y = texts[keep], y[keep]
    reps, folds = (1, 2) if args.quick else (5, 2)
    rows, run = [], 0
    for rep in range(reps):
        skf = StratifiedKFold(folds, shuffle=True, random_state=args.seed + rep)
        for fold, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            run += 1
            print(f"\n=== run {run}/{reps * folds} (repetition {rep + 1}, fold {fold + 1})")
            run_approach(texts[tr], y[tr], texts[te], y[te], args.budget_k, args,
                         args.seed + run, {"run": run, "rep": rep, "fold": fold}, rows)
            pd.DataFrame(rows).to_csv(os.path.join(out, "results_per_run.csv"), index=False)
    df = pd.DataFrame(rows)
    t = summarize(df, ["K"], os.path.join(out, "summary.csv"))
    print("\n=== CLONALG+DE, mean ± std over runs\n" + t.to_string())
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.rcParams.update({"font.family": "serif", "font.size": 10})
        g = df.groupby("n_features")["f1"].agg(["mean", "std"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(g.index, g["mean"], yerr=g["std"], fmt="o-", color="tab:red", capsize=3)
        ax.set_xscale("log")
        ax.set_xticks(g.index)
        ax.set_xticklabels([str(int(k)) for k in g.index])
        ax.set_xlabel("Number of selected features K")
        ax.set_ylabel("F1-score (spam)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out, f"f1_vs_k.{ext}"), dpi=300)
    except ImportError:
        pass


def exp_salman(args):
    out = os.path.join(args.out, "salman_super")
    os.makedirs(out, exist_ok=True)
    print("\n" + "#" * 70 + "\n# PROTOCOL OF SALMAN et al. (2024) - Super SMS, 80/20\n" + "#" * 70)
    texts, y = load_dataset(args.data, dedup=False)
    if args.quick and len(y) > 8000:
        keep = train_test_split(np.arange(len(y)), train_size=8000, stratify=y,
                                random_state=args.seed)[0]
        texts, y = texts[keep], y[keep]
    rows = []
    for seed in args.salman_seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        print(f"\n=== seed {seed} | train={len(tr)} test={len(te)}")
        run_approach(texts[tr], y[tr], texts[te], y[te], args.salman_k, args, seed,
                     {"seed": seed}, rows)
        pd.DataFrame(rows).to_csv(os.path.join(out, "results_per_seed.csv"), index=False)
    t = summarize(pd.DataFrame(rows), ["method"], os.path.join(out, "summary.csv"))
    rep = pd.DataFrame([{"method": f"{m} (reported, Salman et al. 2024)",
                         "accuracy": f"{a:.4f}"} for m, a in SALMAN_REPORTED])
    comp = pd.concat([rep, t.reset_index()[["method", "n_features", "accuracy", "f1", "mcc",
                                             "train_s", "infer_ms"]]], ignore_index=True).fillna("")
    comp.to_csv(os.path.join(out, "comparison_with_salman.csv"), index=False)
    print("\n=== COMPARISON WITH SALMAN et al. (accuracy is the comparable metric)")
    print(comp.to_string(index=False))


def exp_senol(args):
    out = os.path.join(args.out, "senol_uci")
    os.makedirs(out, exist_ok=True)
    print("\n" + "#" * 70 + "\n# PROTOCOL OF SENOL et al. (2026) - UCI, 70/30\n" + "#" * 70)
    texts, y = load_dataset(args.uci, dedup=False)
    if args.prep == "senol":
        texts = senol_preprocess(texts)
    rows, feats = [], {}
    for seed in args.senol_seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
        keep = np.array([len(str(t).strip()) > 0 for t in texts[tr]])
        print(f"\n=== seed {seed}{' (split of Senol et al.)' if seed == 111 else ''} | "
              f"train={keep.sum()} test={len(te)}")
        run_approach(texts[tr][keep], y[tr][keep], texts[te], y[te], args.senol_k, args, seed,
                     {"seed": seed}, rows, feats)
        pd.DataFrame(rows).to_csv(os.path.join(out, "results_per_seed.csv"), index=False)
    df = pd.DataFrame(rows)
    t = summarize(df, ["method"], os.path.join(out, "summary_all_seeds.csv"))
    t111 = summarize(df[df.seed == 111], ["method"], os.path.join(out, "summary_seed111.csv")) \
        if (df.seed == 111).any() else None
    with open(os.path.join(out, "selected_features.json"), "w") as f:
        json.dump(feats, f, indent=1)
    rep = pd.DataFrame([{"source": "Senol et al. 2026 (reported, seed 111)", "method": m,
                         "n_features": k, "accuracy": f"{a:.4f}", "precision": f"{p:.4f}",
                         "recall": f"{r:.4f}", "f1": f"{f:.4f}", "auc": f"{u:.4f}"}
                        for m, k, a, p, r, f, u in SENOL_REPORTED])
    parts = [rep]
    if t111 is not None:
        x = t111.reset_index()
        x.insert(0, "source", "this work (same split, seed 111)")
        parts.append(x)
    x = t.reset_index()
    x.insert(0, "source", f"this work (mean ± std, {len(args.senol_seeds)} splits)")
    parts.append(x)
    comp = pd.concat(parts, ignore_index=True).fillna("")
    comp = comp[["source", "method", "n_features", "accuracy", "precision", "recall", "f1",
                 "auc", "mcc"]]
    comp.to_csv(os.path.join(out, "comparison_with_senol.csv"), index=False)
    print("\n=== COMPARISON WITH SENOL et al. (UCI)")
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(comp.to_string(index=False))


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="CLONALG + DE-weighted ensemble for SMS spam")
    ap.add_argument("--experiment", choices=["budget", "salman", "senol", "all"], default="all")
    ap.add_argument("--data", default="super_sms_dataset.csv", help="Super SMS Dataset CSV")
    ap.add_argument("--uci", default="SMSSpamCollection", help="UCI SMS Spam Collection file")
    ap.add_argument("--out", default="results_clonalg_de")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-k", nargs="+", default=["50", "100", "200", "500", "1000", "2000"])
    ap.add_argument("--salman-k", nargs="+", default=["200", "8000", "all"])
    ap.add_argument("--salman-seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--senol-k", nargs="+", default=["100", "2000", "all"])
    ap.add_argument("--senol-seeds", type=int, nargs="+", default=[111, 1, 2, 3, 4])
    ap.add_argument("--prep", choices=["raw", "senol"], default="raw",
                    help="UCI text: raw, or the preprocessing of Senol et al.")
    ap.add_argument("--max-word", type=int, default=50000)
    ap.add_argument("--max-char", type=int, default=50000)
    ap.add_argument("--pool-factor", type=int, default=3)
    ap.add_argument("--pool-extra", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gen", type=int, default=15)
    ap.add_argument("--sample", type=int, default=15000)
    ap.add_argument("--de-pop", type=int, default=15)
    ap.add_argument("--de-iter", type=int, default=50)
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.out += "_quick"
        args.budget_k, args.salman_k, args.senol_k = ["50", "200"], ["200", "all"], ["100", "all"]
        args.salman_seeds, args.senol_seeds = [42], [111]
        args.gen, args.pop, args.de_iter, args.rf_trees = 4, 8, 15, 50
        args.max_word = args.max_char = 20000

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    t0 = time.time()
    if args.experiment in ("budget", "all"):
        exp_budget(args)
    if args.experiment in ("salman", "all"):
        exp_salman(args)
    if args.experiment in ("senol", "all"):
        exp_senol(args)
    print(f"\nTotal time: {(time.time() - t0) / 60:.1f} min | results in {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

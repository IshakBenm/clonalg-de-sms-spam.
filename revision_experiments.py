#!/usr/bin/env python3
"""
revision_experiments.py — experiments requested by the pre-submission review.
Put it in the same folder as main_clonalg_de.py (it reuses its code).

--task baselines    Re-runs two classical baselines ON YOUR HARDWARE AND YOUR SPLITS:
                      * TF-IDF + linear SVM (TCSVM-BoW style of Salman et al.)
                      * BoW + QuantileTransformer + LR with the preprocessing of Senol et al.
                    Super SMS: stratified 80/20, seeds 42,1,2,3,4 (Salman protocol)
                    UCI:       stratified 70/30, seeds 111,1,2,3,4 (Senol protocol)
                    Paired comparison with your CLONALG+DE results (results_per_seed.csv):
                    mean difference, wins, corrected resampled t-test (Nadeau & Bengio).
--task footprint    Builds DEPLOYABLE models (vectorisers restricted to the selected n-grams,
                    normalised over them only), retrains the ensemble, and measures:
                    accuracy, file size (per component), peak memory, single-message latency.
                    Super SMS, split seed 42: K = 200, 2000 and all features, with and without
                    the random forest (and a small forest), plus the two re-run baselines.
--task sensitivity  One-at-a-time sensitivity of CLONALG (N, G, beta) and DE (F, CR),
                    K = 200, Super SMS split seed 42.
--task all

Usage:
    python revision_experiments.py --task baselines
    python revision_experiments.py --task footprint
    python revision_experiments.py --task sensitivity
"""

import argparse
import copy
import io
import json
import os
import pickle
import time
import tracemalloc
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import LinearSVC

try:
    import main_clonalg_de as M
except ImportError:                      # file saved under another name
    import ais_de_core as M
    assert hasattr(M, "CLONALGSelector"), "main_clonalg_de.py not found"

warnings.filterwarnings("ignore")


# =============================================================================
# 1. BASELINES
# =============================================================================
def tfidf_svm(t_tr, y_tr, t_te, seed):
    t0 = time.perf_counter()
    vec = TfidfVectorizer()
    clf = LinearSVC(C=1.0, random_state=seed).fit(vec.fit_transform(t_tr), y_tr)
    tr = time.perf_counter() - t0
    t1 = time.perf_counter()
    s = clf.decision_function(vec.transform(t_te))
    return s, 0.0, tr, (time.perf_counter() - t1) * 1000 / len(t_te)


def senol_bow_qt_lr(t_tr, y_tr, t_te, seed):
    t_tr, t_te = M.senol_preprocess(t_tr), M.senol_preprocess(t_te)
    t0 = time.perf_counter()
    vec = CountVectorizer()
    qt = QuantileTransformer(n_quantiles=1000, random_state=seed)
    clf = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    clf.fit(qt.fit_transform(vec.fit_transform(t_tr)), y_tr)
    tr = time.perf_counter() - t0
    t1 = time.perf_counter()
    s = clf.predict_proba(qt.transform(vec.transform(t_te)))[:, 1]
    return s, 0.5, tr, (time.perf_counter() - t1) * 1000 / len(t_te)


BASELINES = {"TF-IDF + linear SVM (TCSVM-style)": tfidf_svm,
             "BoW + QuantileTransformer + LR (Senol-style)": senol_bow_qt_lr}


def corrected_ttest(d, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for repeated random splits."""
    d = np.asarray(d, float)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan
    t = d.mean() / np.sqrt((1 / n + n_test / n_train) * var)
    return float(2 * stats.t.sf(abs(t), n - 1))


def run_baselines(args, name, texts, y, test_size, seeds, ours_path, out):
    rows = []
    for seed in seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y,
                                  random_state=seed)
        for bname, fn in BASELINES.items():
            s, thr, tr_s, inf = fn(texts[tr], y[tr], texts[te], seed)
            m = M.evaluate(y[te], s, thr)
            rows.append({"dataset": name, "seed": seed, "method": bname, **m,
                         "train_s": tr_s, "infer_ms": inf})
            print(f"  [{name}] seed {seed:<4} {bname:<45} ACC={m['accuracy']:.4f} "
                  f"F1={m['f1']:.4f} MCC={m['mcc']:.4f}")
    base = pd.DataFrame(rows)
    base.to_csv(os.path.join(out, f"baselines_{name}.csv"), index=False)

    # paired comparison with your results
    if not os.path.isfile(ours_path):
        print(f"  [!] {ours_path} not found: paired comparison skipped")
        return base, None
    ours = pd.read_csv(ours_path)
    n_test = int(round(len(y) * test_size))
    comp = []
    for om in ours["method"].unique():
        a = ours[ours.method == om].set_index("seed")
        for bm in BASELINES:
            b = base[base.method == bm].set_index("seed")
            common = sorted(set(a.index) & set(b.index))
            for metric in ["accuracy", "f1", "mcc"]:
                d = a.loc[common, metric].values - b.loc[common, metric].values
                comp.append({"dataset": name, "ours": om, "baseline": bm, "metric": metric,
                             "ours_mean": a.loc[common, metric].mean(),
                             "baseline_mean": b.loc[common, metric].mean(),
                             "diff": d.mean(), "wins": f"{int((d > 0).sum())}/{len(d)}",
                             "p_corrected_t": corrected_ttest(d, len(y) - n_test, n_test)})
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(out, f"paired_vs_baselines_{name}.csv"), index=False)
    return base, comp


def task_baselines(args):
    out = os.path.join(args.out, "baselines")
    os.makedirs(out, exist_ok=True)
    print("\n### BASELINES RE-RUN ON THE SAME SPLITS AND HARDWARE")
    texts, y = M.load_dataset(args.data, dedup=False)
    b1, c1 = run_baselines(args, "super_salman", texts, y, 0.2, args.salman_seeds,
                           args.ours_salman, out)
    texts, y = M.load_dataset(args.uci, dedup=False)
    b2, c2 = run_baselines(args, "uci_senol", texts, y, 0.3, args.senol_seeds,
                           args.ours_senol, out)
    for b in (b1, b2):
        g = b.groupby(["dataset", "method"])[["accuracy", "precision", "recall", "f1", "mcc",
                                             "train_s", "infer_ms"]].agg(["mean", "std"])
        print("\n" + g.round(4).to_string())
    for c in (c1, c2):
        if c is not None:
            print("\nPaired comparison (ours - baseline):")
            print(c[c.metric == "f1"].round(4).to_string(index=False))


# =============================================================================
# 2. DEPLOYABLE FOOTPRINT
# =============================================================================
class DeployableModel:
    """Vectorisers restricted to the selected n-grams + base classifiers + DE weights."""

    def __init__(self, word_vocab, char_vocab, args, bases=None, rf_trees=None, rf_depth=None):
        kw = dict(sublinear_tf=True, lowercase=True)
        self.word = (TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     token_pattern=r"(?u)\b\w+\b", vocabulary=word_vocab, **kw)
                     if word_vocab else None)
        self.char = (TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     vocabulary=char_vocab, **kw) if char_vocab else None)
        self.args = args
        self.bases = list(bases or M.BASE_NAMES)
        self.rf_trees = rf_trees or args.rf_trees
        self.rf_depth = rf_depth

    def _X(self, texts, fit=False):
        from scipy.sparse import hstack
        parts = []
        for v in (self.word, self.char):
            if v is not None:
                parts.append(v.fit_transform(texts) if fit else v.transform(texts))
        X = hstack(parts).tocsr()
        # renormalise over the selected features only (deployable model)
        norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=1))).ravel()
        norm[norm == 0] = 1.0
        return X.multiply(1 / norm[:, None]).tocsr()

    def fit(self, texts, y, seed):
        X = self._X(texts, fit=True)
        models = M.make_base_models(seed, self.rf_trees)
        models["RF"].set_params(max_depth=self.rf_depth)
        models = {n: models[n] for n in self.bases}
        inner = StratifiedKFold(3, shuffle=True, random_state=seed)
        P = np.column_stack([cross_val_predict(clone(models[n]), X, y, cv=inner,
                                               method="predict_proba")[:, 1]
                             for n in self.bases])
        self.models = {n: clone(models[n]).fit(X, y) for n in self.bases}
        if "RF" in self.models:
            self.models["RF"].set_params(n_jobs=1)   # single-message deployment
        self.w, self.thr = M.de_weights(P, y, self.args, seed)
        return self

    def score(self, texts):
        X = self._X(texts)
        return np.column_stack([self.models[n].predict_proba(X)[:, 1]
                                for n in self.bases]) @ self.w

    def components(self):
        return {"vectorisers": [self.word, self.char], **self.models}

    def n_features(self):
        return sum(len(v.vocabulary) for v in (self.word, self.char) if v is not None)


class BaselineModel:
    """Re-implemented baselines, wrapped for footprint measurement."""

    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def fit(self, texts, y, seed):
        if self.kind == "svm":
            self.vec = TfidfVectorizer()
            self.clf = LinearSVC(C=1.0, random_state=seed).fit(self.vec.fit_transform(texts), y)
            self.thr = 0.0
        else:
            self.vec = CountVectorizer()
            self.qt = QuantileTransformer(n_quantiles=1000, random_state=seed)
            X = self.qt.fit_transform(self.vec.fit_transform(M.senol_preprocess(texts)))
            self.clf = LogisticRegression(max_iter=2000, solver="liblinear",
                                          random_state=seed).fit(X, y)
            self.thr = 0.5
        return self

    def score(self, texts):
        if self.kind == "svm":
            return self.clf.decision_function(self.vec.transform(texts))
        X = self.qt.transform(self.vec.transform(M.senol_preprocess(texts)))
        return self.clf.predict_proba(X)[:, 1]

    def components(self):
        c = {"vectorisers": self.vec, "classifier": self.clf}
        if self.kind != "svm":
            c["scaler"] = self.qt
        return c

    def n_features(self):
        return len(self.vec.vocabulary_)


def size_kb(obj):
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return len(buf.getvalue()) / 1024


def measure(model, t_te, y_te, label, rows):
    m = M.evaluate(y_te, model.score(t_te), model.thr)
    comp = {k: size_kb(v) for k, v in model.components().items()}
    total = size_kb(model)
    blob = pickle.dumps(model)
    tracemalloc.start()
    mdl = pickle.loads(blob)
    mdl.score(t_te[:1])
    peak = tracemalloc.get_traced_memory()[1] / 1024 ** 2
    tracemalloc.stop()
    t0 = time.perf_counter()
    for t in t_te[:500]:
        model.score(np.array([t], dtype=object))
    lat = (time.perf_counter() - t0) * 1000 / 500
    rows.append({"model": label, "n_features": model.n_features(), "accuracy": m["accuracy"],
                 "f1": m["f1"], "mcc": m["mcc"], "size_total_KB": round(total, 1),
                 **{f"size_{k}_KB": round(v, 1) for k, v in comp.items()},
                 "peak_mem_MB": round(peak, 1), "latency_ms_per_msg": round(lat, 3)})
    print(f"  {label:<46} n={model.n_features():<7} F1={m['f1']:.4f} ACC={m['accuracy']:.4f} "
          f"size={total:,.0f} KB  peakMem={peak:.1f} MB  latency={lat:.2f} ms")


def task_footprint(args):
    out = os.path.join(args.out, "footprint")
    os.makedirs(out, exist_ok=True)
    print("\n### DEPLOYABLE FOOTPRINT (Super SMS, 80/20 split, seed 42)")
    texts, y = M.load_dataset(args.data, dedup=False)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                              random_state=args.seed)
    t_tr, t_te, y_tr, y_te = texts[tr], texts[te], y[tr], y[te]
    Xtr, _, _, names = M.vectorize(t_tr, t_te[:10], args.max_word, args.max_char)
    rows = []

    def split_vocab(sel_names):
        w = [n[2:] for n in sel_names if n.startswith("w:")]
        c = [n[2:] for n in sel_names if n.startswith("c:")]
        return ({t: i for i, t in enumerate(sorted(set(w)))} or None,
                {t: i for i, t in enumerate(sorted(set(c)))} or None)

    def select(K):
        P = max(args.pool_factor * K, K + args.pool_extra)
        pool, sc = M.chi2_pool(Xtr, y_tr, P)
        clg = M.CLONALGSelector(K, pop_size=args.pop, n_gen=args.gen, alpha=args.alpha,
                                sample=args.sample, seed=args.seed).fit(Xtr, y_tr, pool, sc)
        sel = names[np.sort(clg.selected_)]
        with open(os.path.join(out, f"selected_K{K}.json"), "w") as f:
            json.dump(sel.tolist(), f, indent=1)
        return split_vocab(sel)

    no_rf = ["NB", "LR", "SVM"]
    v200, v2000, vall = select(200), select(2000), split_vocab(names)
    variants = [
        ("K=200, NB+LR+SVM+RF(200)", v200, {}),
        ("K=200, NB+LR+SVM+RF(20, depth 20)", v200, {"rf_trees": 20, "rf_depth": 20}),
        ("K=200, NB+LR+SVM (no RF)", v200, {"bases": no_rf}),
        ("K=2000, NB+LR+SVM (no RF)", v2000, {"bases": no_rf}),
        ("All features, NB+LR+SVM (no RF)", vall, {"bases": no_rf}),
        ("All features, NB+LR+SVM+RF(200)", vall, {}),
    ]
    for label, (wv, cv), kw in variants:
        measure(DeployableModel(wv, cv, args, **kw).fit(t_tr, y_tr, args.seed), t_te, y_te,
                label, rows)
    for kind, label in (("svm", "Baseline TF-IDF + linear SVM"),
                        ("qt", "Baseline BoW + QuantileTransformer + LR")):
        measure(BaselineModel(kind, args.seed).fit(t_tr, y_tr, args.seed), t_te, y_te, label, rows)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "footprint.csv"), index=False)
    print("\n" + df.to_string(index=False))


# =============================================================================
# 3. SENSITIVITY
# =============================================================================
def task_sensitivity(args):
    out = os.path.join(args.out, "sensitivity")
    os.makedirs(out, exist_ok=True)
    print("\n### SENSITIVITY (K = 200, Super SMS, 80/20 split, seed 42)")
    texts, y = M.load_dataset(args.data, dedup=False)
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                              random_state=args.seed)
    Xtr, Xte, _, _ = M.vectorize(texts[tr], texts[te], args.max_word, args.max_char)
    y_tr, y_te = y[tr], y[te]
    K = 200
    P = max(args.pool_factor * K, K + args.pool_extra)
    pool, sc = M.chi2_pool(Xtr, y_tr, P)
    default = dict(pop=12, gen=15, beta=0.15, F=0.5, CR=0.9)
    grid = [("default", {})] + \
           [(f"N={v}", {"pop": v}) for v in (8, 20)] + \
           [(f"G={v}", {"gen": v}) for v in (10, 25)] + \
           [(f"beta={v}", {"beta": v}) for v in (0.10, 0.25)] + \
           [(f"F={v}", {"F": v}) for v in (0.3, 0.8)] + \
           [(f"CR={v}", {"CR": v}) for v in (0.7, 1.0)]
    cache, rows = {}, []
    for label, ch in grid:
        p = {**default, **ch}
        key = (p["pop"], p["gen"], p["beta"])
        t0 = time.time()
        if key not in cache:
            clg = M.CLONALGSelector(K, pop_size=p["pop"], n_gen=p["gen"], beta=p["beta"],
                                    alpha=args.alpha, sample=args.sample,
                                    seed=args.seed).fit(Xtr, y_tr, pool, sc)
            cache[key] = (np.sort(clg.selected_), time.time() - t0)
        sel, t_sel = cache[key]
        a = copy.copy(args)
        orig = M.differential_evolution

        def de_patch(*aa, **kw):
            kw["mutation"], kw["recombination"] = p["F"], p["CR"]
            return orig(*aa, **kw)
        M.differential_evolution = de_patch
        try:
            s, thr, w, tr_s, _ = M.de_ensemble(Xtr[:, sel], y_tr, Xte[:, sel], a, args.seed)
        finally:
            M.differential_evolution = orig
        m = M.evaluate(y_te, s, thr)
        rows.append({"setting": label, **p, "f1": m["f1"], "accuracy": m["accuracy"],
                     "mcc": m["mcc"], "selection_s": round(t_sel, 1), "total_s": round(t_sel + tr_s, 1)})
        print(f"  {label:<10} F1={m['f1']:.4f} ACC={m['accuracy']:.4f}  ({t_sel + tr_s:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "sensitivity.csv"), index=False)
    print(f"\nF1 range over all settings: {df.f1.min():.4f} – {df.f1.max():.4f} "
          f"(default {df.f1.iloc[0]:.4f})")


# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["baselines", "footprint", "sensitivity", "all"], default="all")
    ap.add_argument("--data", default="super_sms_dataset.csv")
    ap.add_argument("--uci", default="SMSSpamCollection")
    ap.add_argument("--ours-salman", default="results_clonalg_de/salman_super/results_per_seed.csv")
    ap.add_argument("--ours-senol", default="results_clonalg_de/senol_uci/results_per_seed.csv")
    ap.add_argument("--out", default="results_revision")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--salman-seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--senol-seeds", type=int, nargs="+", default=[111, 1, 2, 3, 4])
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
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.task in ("baselines", "all"):
        task_baselines(args)
    if args.task in ("footprint", "all"):
        task_footprint(args)
    if args.task in ("sensitivity", "all"):
        task_sensitivity(args)


if __name__ == "__main__":
    main()
"""Aggregate per-query results over seeds: means with 95% t-intervals across seeds, paired bootstrap tests."""
import glob, json, random
from collections import defaultdict
import numpy as np

T4 = 2.776  # t(0.975, df=4) for five seeds
SIZES = [500, 2000, 8000]
runs = {}
for f in sorted(glob.glob("results/p*_s*.json")):
    d = json.load(open(f))
    runs[(d["meta"]["parties"], d["meta"]["seed"])] = d


def ci(vals):
    v = np.array(vals, float)
    return float(v.mean()), float(T4 * v.std(ddof=1) / np.sqrt(len(v)))


out = {"meta": {}, "table": {}, "by_type": {}, "router": {}, "paired": {}}
for p in SIZES:
    metas = [runs[(p, s)]["meta"] for s in range(1, 6)]
    out["meta"][p] = {k: float(np.mean([m[k] for m in metas])) for k in
                      ["nodes", "edges", "leiden_communities", "leiden_modularity", "share_communities_multi_context",
                       "mean_contexts_per_community", "tau", "lambda"]}
    per = defaultdict(lambda: defaultdict(list))
    rows_by = defaultdict(list)
    for s in range(1, 6):
        by_m = defaultdict(list)
        for r in runs[(p, s)]["rows"]:
            by_m[r["method"]].append(r)
            rows_by[r["method"]].append(r)
        for m, rs in by_m.items():
            per[m]["hit1"].append(np.mean([r["hit1"] for r in rs]))
            per[m]["hit5"].append(np.mean([r["hit5"] for r in rs]))
            per[m]["explored"].append(np.mean([r["explored"] for r in rs]))
            per[m]["leak"].append(np.mean([r["unauth"] > 0 for r in rs]))
            per[m]["unauth"].append(np.mean([r["unauth"] for r in rs]))
            per[m]["ctx"].append(np.mean([r["n_ctx"] for r in rs]))
            per[m]["hit1_single"].append(np.mean([r["hit1"] for r in rs if not r["cross"]]))
            per[m]["hit1_cross"].append(np.mean([r["hit1"] for r in rs if r["cross"]]))
            if rs[0]["route_exact"] is not None:
                per[m]["route_exact"].append(np.mean([r["route_exact"] for r in rs]))
                per[m]["route_recall"].append(np.mean([r["route_recall"] for r in rs]))
                per[m]["route_recall_unseen"].append(np.mean([r["route_recall"] for r in rs if r["form"] == 2]))
                per[m]["route_recall_seen"].append(np.mean([r["route_recall"] for r in rs if r["form"] < 2]))
    out["table"][p] = {m: {k: ci(v) for k, v in d.items()} for m, d in per.items()}

    # paired bootstrap over pooled test queries (same queries for every method)
    def paired(a, b, key="hit1", cross=None, B=2000):
        ra = [r for r in rows_by[a] if cross is None or r["cross"] == cross]
        rb = [r for r in rows_by[b] if cross is None or r["cross"] == cross]
        x = np.array([r[key] for r in ra], float) - np.array([r[key] for r in rb], float)
        rng = np.random.default_rng(0)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
        return {"diff": float(x.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "n": len(x)}
    out["paired"][p] = {
        "acl_vs_mono": paired("acl_contexts", "monolithic"),
        "oracle_vs_mono": paired("ddd_oracle", "monolithic"),
        "oracle_vs_acl": paired("ddd_oracle", "acl_contexts"),
        "learned_vs_acl": paired("ddd_learned", "acl_contexts"),
        "leiden3_vs_acl": paired("leiden_top3", "acl_contexts"),
        "leiden3acl_vs_acl": paired("leiden_top3_acl", "acl_contexts"),
        "oracle_vs_oracle_noid_cross": paired("ddd_oracle", "ddd_oracle_no_identity", cross=True),
        "oracle_vs_oracle_noid_single": paired("ddd_oracle", "ddd_oracle_no_identity", cross=False),
    }
out["restricted"] = {}
for p in SIZES:
    res = {}
    pooled = defaultdict(list)
    for s in range(1, 6):
        by = defaultdict(list)
        for r in runs[(p, s)]["restricted_rows"]:
            by[(r["method"], r["restricted"])].append(r)
            by[(r["method"], "tpl:" + r["template"])].append(r)
            pooled[(r["method"], r["restricted"])].append(r)
        for k, rs in by.items():
            res.setdefault(k, defaultdict(list))
            res[k]["disclosed1"].append(np.mean([r["disclosed1"] for r in rs]))
            res[k]["disclosed5"].append(np.mean([r["disclosed5"] for r in rs]))
            res[k]["leak"].append(np.mean([r["unauth"] > 0 for r in rs]))
            res[k]["n"].append(len(rs))
    o = {f"{m}|{kind}": {k: ci(v) for k, v in d.items()} for (m, kind), d in res.items()}
    x = np.array([r["disclosed1"] for r in pooled[("post_filter", "evidence")]], float) - \
        np.array([r["disclosed1"] for r in pooled[("acl_contexts", "evidence")]], float)
    rng = np.random.default_rng(0)
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    o["excess_postfilter_vs_acl_evidence"] = {"diff": float(x.mean()), "lo": float(np.percentile(bs, 2.5)),
                                              "hi": float(np.percentile(bs, 97.5)), "n": len(x)}
    out["restricted"][p] = o
json.dump(out, open("summary.json", "w"), indent=1)
for p in SIZES:
    print(f"\n== restricted {p}")
    for k, v in sorted(out["restricted"][p].items()):
        if "disclosed1" in v:
            print(f"  {k:36s} d1 {v['disclosed1'][0]:.3f}±{v['disclosed1'][1]:.3f} d5 {v['disclosed5'][0]:.3f} leak {v['leak'][0]:.3f} n/seed {v['n'][0]:.0f}")
        else:
            print("  ", k, v)

for p in SIZES:
    print(f"\n== {p} parties  meta {json.dumps({k: round(v, 3) for k, v in out['meta'][p].items()})}")
    for m, d in out["table"][p].items():
        f = lambda k: f"{d[k][0]:.3f}±{d[k][1]:.3f}"
        extra = f"  route exact {f('route_exact')} recall {f('route_recall')} seen {f('route_recall_seen')} unseen {f('route_recall_unseen')}" if "route_exact" in d else ""
        print(f"{m:24s} h1 {f('hit1')} h5 {f('hit5')} single {f('hit1_single')} cross {f('hit1_cross')} expl {d['explored'][0]:6.1f} leak {f('leak')} unauth {d['unauth'][0]:5.1f} ctx {d['ctx'][0]:.2f}{extra}")
    for k, v in out["paired"][p].items():
        print(f"  {k:30s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] n={v['n']}")

"""Run every retrieval method on one benchmark instance (size, seed) and write per-query results."""
import argparse, json, random, time
from collections import deque
from pathlib import Path

import numpy as np
import igraph as ig
import leidenalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression

from generator import generate, make_queries, make_restricted_queries, CONTEXTS, IDENTITY
import scipy.sparse as sp

HOPS, ANCHORS, HUB_CAP = 3, 5, 50


def build(n_parties, seed, n_queries):
    g, parties = generate(n_parties, seed)
    qs = make_queries(parties, n_queries, seed)
    n = len(g.nodes)
    adj = [[] for _ in range(n)]
    for u, v, _ in g.edges:
        adj[u].append(v); adj[v].append(u)
    ctx = np.array([nd["ctx"] for nd in g.nodes])
    typ = np.array([nd["type"] for nd in g.nodes])
    vec = TfidfVectorizer(sublinear_tf=True, token_pattern=r"(?u)\b[\w-]+\b")
    X = normalize(vec.fit_transform([nd["text"] for nd in g.nodes]))
    rq = make_restricted_queries(parties, 300, seed, start_qid=len(qs))
    Q = normalize(vec.transform([q["text"] for q in qs + rq]))
    return g, qs, rq, adj, ctx, typ, vec, X, Q


def leiden_partition(g, seed):
    G = ig.Graph(n=len(g.nodes), edges=[(u, v) for u, v, _ in g.edges])
    part = leidenalg.find_partition(G, leidenalg.ModularityVertexPartition, seed=seed)
    return np.array(part.membership), part.modularity


def retrieve(qi, q, scope, adj, typ, sims, lam=0.1, cand_ok=None):
    """Anchor on the top-scoring in-scope nodes, expand HOPS hops inside scope from each anchor,
    and rank target-type nodes by own text match plus anchor evidence decayed by path length."""
    s = sims.copy()
    s[~scope] = -1
    anchors = [a for a in np.argpartition(-s, ANCHORS)[:ANCHORS] if s[a] > 0]
    support = {}
    for a in anchors:
        dist = {a: 0}
        dq = deque([a])
        while dq:
            u = dq.popleft()
            if dist[u] >= HOPS or (len(adj[u]) > HUB_CAP and dist[u] > 0):
                continue
            for v in adj[u]:
                if scope[v] and v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        for v, d in dist.items():
            support[v] = max(support.get(v, 0.0), sims[a] * 0.5 ** d)
    visited = np.fromiter(support.keys(), dtype=int, count=len(support))
    cands = [v for v in visited if typ[v] == q["type"] and (cand_ok is None or cand_ok[v])]
    ranked = sorted(cands, key=lambda v: -(sims[v] + lam * support[v]))
    rank = ranked.index(q["target"]) + 1 if q["target"] in ranked else None
    return visited, rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parties", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--queries", type=int, default=700)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    t0 = time.time()
    g, qs, rq, adj, ctx, typ, vec, X, Q = build(a.parties, a.seed, a.queries)
    n = len(g.nodes)
    dev, test = qs[: len(qs) * 2 // 7], qs[len(qs) * 2 // 7:]

    # bounded-context router: query vs. context profile (centroid of context node texts)
    prof = normalize(np.vstack([np.asarray(X[ctx == c].mean(axis=0)) for c in CONTEXTS]))
    rsim = np.asarray(Q @ prof.T)

    def route(qi, tau):
        r = rsim[qi]
        return {c for c, v in zip(CONTEXTS, r) if v >= tau * r.max()}

    # tune tau on dev: maximise exact-set routing accuracy (gold + CUSTOMER kernel ignored)
    best = max([0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               key=lambda t: np.mean([route(q["qid"], t) - {"CUSTOMER"} == set(q["gold"]) for q in dev]))

    # learned router: one logistic regression per context, trained on dev queries written in
    # paraphrase forms 0 and 1 only; form 2 is never seen in training
    train = [q for q in dev if q["form"] < 2]
    clfs = {}
    for c in CONTEXTS[1:]:
        y = [c in q["gold"] for q in train]
        clfs[c] = LogisticRegression(C=10, max_iter=2000).fit(Q[[q["qid"] for q in train]], y)

    def route_learned(qi):
        return {c for c, m in clfs.items() if m.predict_proba(Q[qi])[0, 1] >= 0.3}

    # ranking weight: tuned once on dev with the monolithic scope, then fixed for every method
    def dev_hit(lam):
        hits = 0
        for q in dev:
            sims = (X @ Q[q["qid"]].T).toarray().ravel()
            hits += retrieve(q["qid"], q, np.ones(n, bool), adj, typ, sims, lam)[1] == 1
        return hits
    lam = max([0.0, 0.25, 0.5, 1.0, 2.0, 4.0], key=dev_hit)

    memb, modularity = leiden_partition(g, a.seed)
    ncom = memb.max() + 1
    cdoc = normalize(np.vstack([np.asarray(X[memb == k].mean(axis=0)) for k in range(ncom)]))
    csim = np.asarray(Q @ cdoc.T)
    # how organizational and topological boundaries disagree: share of communities spanning >1 context
    spans = [len(set(ctx[memb == k]) - {IDENTITY}) for k in range(ncom)]

    rng = random.Random(a.seed)
    rows = []
    for q in test:
        qi = q["qid"]
        sims = (X @ Q[qi].T).toarray().ravel()
        auth = set(q["authorized"])
        kernel = ctx == IDENTITY
        routed = route(qi, best) & auth
        learned = route_learned(qi) & auth
        k = max(1, len(routed - {"CUSTOMER"}))
        rand = set(rng.sample(sorted(auth - {"CUSTOMER"}), min(k, len(auth - {"CUSTOMER"}))))
        scopes = {
            "monolithic": np.ones(n, bool),
            "acl_contexts": np.isin(ctx, list(auth)) | kernel,
            "ddd_routed": np.isin(ctx, list(routed | {"CUSTOMER"})) | kernel,
            "ddd_routed_no_identity": np.isin(ctx, list(routed | {"CUSTOMER"})),
            "ddd_learned": np.isin(ctx, list(learned | {"CUSTOMER"})) | kernel,
            "ddd_random": np.isin(ctx, list(rand | {"CUSTOMER"})) | kernel,
            "ddd_oracle": np.isin(ctx, list(set(q["gold"]) | {"CUSTOMER"})) | kernel,
            "ddd_oracle_no_identity": np.isin(ctx, list(set(q["gold"]) | {"CUSTOMER"})),
            "leiden_top1": np.isin(memb, np.argsort(-csim[qi])[:1]),
            "leiden_top3": np.isin(memb, np.argsort(-csim[qi])[:3]),
        }
        scopes["leiden_top3_acl"] = scopes["leiden_top3"] & scopes["acl_contexts"]
        scopes["post_filter"] = scopes["monolithic"]
        for m, sc in scopes.items():
            ok = scopes["acl_contexts"] if m == "post_filter" else None
            visited, rank = retrieve(qi, q, sc, adj, typ, sims, lam, ok)
            unauth = int(np.sum(~np.isin(ctx[visited], list(auth | {IDENTITY}))))
            rows.append({"method": m, "qid": qi, "template": q["template"], "cross": q["cross"],
                         "hit1": rank == 1, "hit5": rank is not None and rank <= 5, "found": rank is not None,
                         "explored": int(len(visited)), "unauth": unauth,
                         "form": q["form"], "n_ctx": len(set(ctx[visited]) - {IDENTITY}),
                         "route_exact": ((route(qi, best) - {"CUSTOMER"}) == set(q["gold"]) if m == "ddd_routed" else
                                         route_learned(qi) == set(q["gold"]) if m == "ddd_learned" else None),
                         "route_recall": (set(q["gold"]) <= route(qi, best) if m == "ddd_routed" else
                                          set(q["gold"]) <= route_learned(qi) if m == "ddd_learned" else None)})
    # requests the role may not answer: any correct answer here is a disclosure
    rrows = []
    for q in rq:
        qi = q["qid"]
        sims = (X @ Q[qi].T).toarray().ravel()
        auth = set(q["authorized"])
        kernel = ctx == IDENTITY
        acl = np.isin(ctx, list(auth)) | kernel
        learned = route_learned(qi) & auth
        scopes = {
            "monolithic": (np.ones(n, bool), None),
            "post_filter": (np.ones(n, bool), acl),
            "acl_contexts": (acl, None),
            "ddd_learned": (np.isin(ctx, list(learned | {"CUSTOMER"})) | kernel, None),
            "leiden_top3": (np.isin(memb, np.argsort(-csim[qi])[:3]), None),
            "leiden_top3_acl": (np.isin(memb, np.argsort(-csim[qi])[:3]) & acl, None),
        }
        for m, (sc, ok) in scopes.items():
            visited, rank = retrieve(qi, q, sc, adj, typ, sims, lam, ok)
            unauth = int(np.sum(~np.isin(ctx[visited], list(auth | {IDENTITY}))))
            n_cands = int(sum(1 for v in visited if typ[v] == q["type"] and (ok is None or ok[v])))
            rrows.append({"method": m, "qid": qi, "template": q["template"], "restricted": q["restricted"],
                          "disclosed1": rank == 1, "disclosed5": rank is not None and rank <= 5,
                          "unauth": unauth, "n_cands": n_cands})

    meta = {"parties": a.parties, "seed": a.seed, "nodes": n, "edges": len(g.edges), "queries_dev": len(dev),
            "queries_test": len(test), "tau": best, "lambda": lam, "leiden_communities": int(ncom), "leiden_modularity": modularity,
            "share_communities_multi_context": float(np.mean([s > 1 for s in spans])),
            "mean_contexts_per_community": float(np.mean(spans)), "seconds": round(time.time() - t0, 1)}
    out = Path(a.out); out.mkdir(exist_ok=True)
    (out / f"p{a.parties}_s{a.seed}.json").write_text(json.dumps({"meta": meta, "rows": rows, "restricted_rows": rrows}))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()

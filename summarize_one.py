import json,sys,statistics as s,collections
d=json.load(open(sys.argv[1]));print(d['meta'])
agg=collections.defaultdict(list)
for r in d['rows']: agg[r['method']].append(r)
for m,rs in agg.items():
  extra=''
  if rs[0]['route_exact'] is not None:
    extra=f" route_exact {s.mean(r['route_exact'] for r in rs):.2f} recall {s.mean(r['route_recall'] for r in rs):.2f} unseen_recall {s.mean(r['route_recall'] for r in rs if r['form']==2):.2f}"
  print(f"{m:24s} h1 {s.mean(r['hit1'] for r in rs):.3f} h5 {s.mean(r['hit5'] for r in rs):.3f} expl {s.mean(r['explored'] for r in rs):7.1f} leak {s.mean(r['unauth']>0 for r in rs):.3f} xh1 {s.mean(r['hit1'] for r in rs if r['cross']):.3f}"+extra)

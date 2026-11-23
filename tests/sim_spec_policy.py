"""Online policy replay on fixed-8 traces. Plain segments skip ahead in the trace by the
number of tokens they emit (rounds covering that many tokens are dropped).

    python tests/sim_spec_policy.py TRACE.json [draft_ms] [overhead_ms]   (TRACE.json from tests/trace_spec.py)
"""

import json
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from inkling2_mini.speculative import BlockPolicy  # noqa: E402

d = json.load(open(sys.argv[1]))
R = [r for r in d["rounds"] if len(r["probs"]) == 7]  # full rounds (not cut by max_tokens)
P = d["plain_ms"] / 1e3
V = {int(k): v / 1e3 for k, v in d["verify_ms"].items()}
D = float(sys.argv[2]) / 1e3 if len(sys.argv) > 2 else np.mean([r["draft_s"] for r in R])
O = float(sys.argv[3]) / 1e3 if len(sys.argv) > 3 else np.mean([r["round_s"] - r["draft_s"] for r in R]) - V[8]
BUBBLE = 0.002  # a gated round that falls back to plain: draft + sync, then the plain loop
cost = {L: D + V[L] + O for L in range(2, 9)}
by_run = defaultdict(list)
for r in R:
    by_run[r["run"]].append(r)
EDGES = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995])


def table(rs):
    n = np.zeros(len(EDGES) + 1); a = np.zeros(len(EDGES) + 1)
    for r in rs:
        for i in range(min(r["acc"] + 1, 7)):
            b = np.searchsorted(EDGES, r["probs"][i])
            n[b] += 1; a[b] += i < r["acc"]
    return n, a


class Calib:
    def __init__(self, n, a, strength=20.0, online=True):
        tot = n.sum()
        self.n = n / tot * strength * len(n) if tot else np.ones_like(n)
        self.a = a / tot * strength * len(n) if tot else self.n * 0.5
        self.online = online

    def __call__(self, probs):
        b = np.searchsorted(EDGES, probs)
        return (self.a[b] + 0.5) / (self.n[b] + 1)

    def update(self, probs, acc):
        if not self.online:
            return
        for i in range(min(acc + 1, len(probs))):
            b = np.searchsorted(EDGES, probs[i])
            self.n[b] += 1; self.a[b] += i < acc


def best_L(c, lam):
    vals, surv, e = {}, 1.0, 1.0
    for L in range(2, 9):
        surv *= c[L - 2]; e += surv
        vals[L] = e - lam * (V[L] + O)
    L = max(vals, key=vals.get)
    return (L, vals[L]) if vals[L] > 0 else (1, 0.0)


def replay(rs, step):
    """step(state, round) -> (tokens, seconds, skip_tokens). Returns tok/s."""
    tok = t = 0.0
    i, st = 0, {}
    while i < len(rs):
        k, s, skip = step(st, rs[i])
        tok += k; t += s
        i += 1
        while skip > 0 and i < len(rs):
            skip -= rs[i]["acc"] + 1; i += 1
    return tok / t


def current_policy(rs):
    pol = BlockPolicy(8)
    pol.cost = {1: P, **cost}; pol.prior = dict(pol.cost)

    def step(st, r):
        tok = t = 0.0
        bs = pol.choose()
        if bs <= 1:
            k = pol.plain_segment
            pol.observe_plain(P)
            return k, k * P, k - 1  # this round's tokens are part of the segment
        acc = min(r["acc"], bs - 1)
        pol.observe(bs, acc, cost[bs])
        return acc + 1, cost[bs], 0

    return replay(rs, step)


def gated(rs, calib, dinkelbach=True, backoff=True, seg0=24, seg_max=256):
    st0 = dict(seg=seg0, lam_tok=0.0, lam_t=0.0)

    def step(st, r):
        if not st:
            st.update(st0)
        lam = 1 / P
        if dinkelbach and st["lam_t"] > 0:
            lam = max(lam, st["lam_tok"] / st["lam_t"])
        c = calib(np.array(r["probs"]))
        L, _ = best_L(c, lam)
        if L == 1:
            seg = st["seg"]
            if backoff:
                st["seg"] = min(2 * seg, seg_max)
            k, s = seg, D + BUBBLE + seg * P
        else:
            acc = min(r["acc"], L - 1)
            calib.update(r["probs"][: L - 1], acc)
            k, s = acc + 1, cost[L]
            if acc >= 1:
                st["seg"] = seg0
        st["lam_tok"] = 0.9 * st["lam_tok"] + k
        st["lam_t"] = 0.9 * st["lam_t"] + s
        return k, s, (k - 1 if L == 1 else 0)

    return replay(rs, step)


print(f"P {P*1e3:.1f}  D {D*1e3:.1f}  O {O*1e3:.1f} ms")
cols = ["plain", "current", "gate", "gate+bo", "gate+bo+dk", "gate+bo+dk offline-calib"]
print(f"{'run':14s} " + " ".join(f"{c:>12s}" for c in cols))
agg = defaultdict(list)
for run, rs in sorted(by_run.items()):
    others = [r for r in R if r["run"].split("@")[0] != run.split("@")[0]]
    n, a = table(others)
    vals = [1 / P, current_policy(rs),
            gated(rs, Calib(n, a), dinkelbach=False, backoff=False),
            gated(rs, Calib(n, a), dinkelbach=False, backoff=True),
            gated(rs, Calib(n, a), dinkelbach=True, backoff=True),
            gated(rs, Calib(n, a, online=False), dinkelbach=True, backoff=True)]
    print(f"{run:14s} " + " ".join(f"{v:12.1f}" for v in vals))
    for c, v in zip(cols, vals):
        agg[c].append(v)
print(f"{'geomean':14s} " + " ".join(f"{np.exp(np.mean(np.log(agg[c]))):12.1f}" for c in cols))

n, a = table(R)
print("\ncalibration (all):", " ".join(f"<{e}:{(aa / nn if nn else 0):.2f}" for e, nn, aa in zip(list(EDGES) + [1.0], n, a)))
for temp in ("T0", "T1"):
    n, a = table([r for r in R if r["run"].endswith(temp)])
    print(f"calibration ({temp}):", " ".join(f"{(aa / nn if nn else 0):.2f}" for nn, aa in zip(n, a)))

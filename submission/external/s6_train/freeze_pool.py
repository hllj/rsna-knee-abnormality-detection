"""Freeze the S6F0 pool from the anchor precompute manifest (CPU, seconds).

pool = every study of precompute_order.json whose precompute succeeded (anchor file present, rendered rows > 0, >= 1 train
candidate and >= 1 stride-1 eval window).  The trainer's `done` set is this list (train_s6f0.py --pool); the resume check needs it
frozen, so the real run reads data/pool_ok.json written ONCE here with --final (refuses while the precompute is incomplete).
Also writes data/pool_ok.counts.json: per-group counts (train / fold-0 / gold), the studies that failed, and train-candidate stats.
--smoke writes data/pool_smoke.json from whatever has finished so far (for the smokes; never used by the real run)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s6common as SC
from precompute_anchors import read_manifest


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--final", action="store_true")
    g.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    order = SC.load_json("precompute_order.json")
    train, f0, gold = set(SC.load_json("train_ids.json")), set(SC.load_json("fold0_ondisk.json")), set(SC.load_json("gold58.json"))
    man = read_manifest(os.path.join(SC.ANCHORS, "manifest.jsonl"))
    missing = [u for u in order if u not in man]
    if a.final and missing:
        raise SystemExit("precompute incomplete: %d of %d studies not finished" % (len(missing), len(order)))
    ok, bad = [], {}
    for u in order:
        r = man.get(u)
        if r is None:
            continue
        good = (r.get("error") is None and os.path.exists(SC.anchor_path(u)) and r.get("rows", 0) > 0
                and r.get("train_cands", 0) >= 1 and r.get("eval1_windows", 0) >= 1)
        if good:
            ok.append(u)
        else:
            bad[u] = r.get("error") or "empty render"
    ok = sorted(ok)
    okset = set(ok)
    tc = [man[u]["train_cands"] for u in ok if u in train]
    counts = {"when": SC.amman_now().isoformat(timespec="seconds"), "final": bool(a.final), "pool": len(ok),
              "train": len(okset & train), "fold0": len(okset & f0), "gold": len(okset & gold),
              "train_expected": len(train), "fold0_expected": len(f0), "gold_expected": len(gold), "not_finished": len(missing),
              "failed": bad, "failed_by_group": {"train": sum(u in train for u in bad), "fold0": sum(u in f0 for u in bad),
                                                 "gold": sum(u in gold for u in bad)},
              "train_cands_min": int(min(tc)) if tc else None, "train_cands_lt64": int(sum(t < 64 for t in tc)),
              "eval1_windows_fold0_sum": int(sum(man[u]["eval1_windows"] for u in ok if u in f0)),
              "eval1_windows_gold_sum": int(sum(man[u]["eval1_windows"] for u in ok if u in gold)),
              "train_windows_per_epoch_at_64": int(sum(min(64, t) for t in tc))}
    name = "pool_ok" if a.final else "pool_smoke"
    path = os.path.join(SC.DATA_DIR, name + ".json")
    if a.final and os.path.exists(path):
        if json.load(open(path)) != ok:
            raise SystemExit("%s exists and differs: the pool is frozen (delete it only before training has started)" % path)
        print("pool already frozen and identical:", path)
    SC.atomic_json(ok, path)
    SC.atomic_json(counts, os.path.join(SC.DATA_DIR, name + ".counts.json"), indent=1)
    print(json.dumps({k: v for k, v in counts.items() if k != "failed"}))


if __name__ == "__main__":
    main()

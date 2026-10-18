"""S6F0 data files (CPU, seconds).  Writes, atomically, under S6F0/data/:
  targets_s6f0.csv     compact_train --targets format: StudyInstanceUID, is_gold, review_holdout (0), fold (folds_v0; gold -1),
                       L|prior = labels_v1_blendjev (blank for gold), L|strict (blank everywhere) for the 12 LABELS
  train_ids.json       folds_v0 fold in {1,2,3,4}, in blendjev, DICOM study dir on disk, not gold          (expect 3,408)
  fold0_ondisk.json    folds_v0 fold 0 with the DICOM study dir on disk                                    (expect 855 of 870)
  fold0_all.json       folds_v0 fold 0 (870)
  gold58.json          folds_v0 fold -1 = raw/train.csv rows with all 12 labels (58), on disk
  smoke_train.json     24 seeded train studies (the smokes), smoke_ho.json 4 fold-0 + 2 gold studies
  precompute_order.json  every study the run needs (train + fold-0 on disk + gold), smoke studies first
No label beyond blendjev is read; train.csv is read only to cross-check the gold ids."""
import csv, io, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s6common as SC

R = "/home/danial/rsna-knee"
FOLDS = f"{R}/public/raptor_train/folds_v0.json"
BLEND = f"{R}/public/raptor_train/labels_v1_blendjev.parquet"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
          "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]          # = compact_train.LABELS (ct:21-22)


def main():
    os.makedirs(SC.DATA_DIR, exist_ok=True)
    folds = json.load(open(FOLDS))["folds"]
    bj = pd.read_parquet(BLEND)
    assert list(bj.columns) == ["StudyInstanceUID"] + LABELS, list(bj.columns)
    assert bj["StudyInstanceUID"].is_unique and np.isfinite(bj[LABELS].values).all()
    bj = bj.set_index("StudyInstanceUID")
    on_disk = set(os.listdir(os.path.join(SC.DATA, "train_series")))
    deleted = {d["study"] for d in json.load(open(f"{R}/raw/DELETED_2026-09-29.json"))["studies"]}
    assert not (deleted & on_disk)
    tr_csv = pd.read_csv(f"{R}/raw/train.csv")
    gold_csv = set(tr_csv.loc[tr_csv[LABELS].notna().all(axis=1), "StudyInstanceUID"])
    gold = sorted(u for u, f in folds.items() if f == -1)
    assert set(gold) == gold_csv and len(gold) == 58 and not (set(gold) & set(bj.index))
    gold_on = [u for u in gold if u in on_disk]
    f0_all = sorted(u for u, f in folds.items() if f == 0)
    f0 = [u for u in f0_all if u in on_disk]
    train = sorted(u for u, f in folds.items() if f in (1, 2, 3, 4) and u in bj.index and u in on_disk and u not in gold_csv)
    lab_missing = sorted(u for u, f in folds.items() if f >= 0 and u not in bj.index)
    print("folds_v0: %d studies; fold0 %d (on disk %d); train folds 1-4 on disk + labelled %d; gold %d (on disk %d); "
          "folds>=0 without a blendjev row %d" % (len(folds), len(f0_all), len(f0), len(train), len(gold), len(gold_on), len(lab_missing)))
    per_fold = {k: sum(1 for u in train if folds[u] == k) for k in (1, 2, 3, 4)}
    print("train per fold:", per_fold)
    assert len(f0_all) == 870 and len(f0) == 855 and len(train) == 3408 and len(gold_on) == 58, "counts differ from the maps"

    # targets in the compact_train format (ct:637-652): every labelled study (all folds) + gold rows with blank labels
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["StudyInstanceUID", "is_gold", "review_holdout", "fold"] + [L + "|prior" for L in LABELS] + [L + "|strict" for L in LABELS])
    for u in sorted(bj.index):
        v = bj.loc[u, LABELS].values.astype(np.float64)
        w.writerow([u, 0, 0, int(folds[u])] + [repr(float(x)) for x in v] + [""] * len(LABELS))
    for u in gold:
        w.writerow([u, 1, 0, -1] + [""] * (2 * len(LABELS)))
    tmp = os.path.join(SC.DATA_DIR, "targets_s6f0.csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(buf.getvalue())
    os.replace(tmp, os.path.join(SC.DATA_DIR, "targets_s6f0.csv"))

    rng = np.random.default_rng(20261010)
    smoke_train = sorted(rng.choice(train, 24, replace=False).tolist())
    smoke_ho = sorted(rng.choice(f0, 4, replace=False).tolist()) + gold_on[:2]
    rest = [u for u in train + f0 + gold_on if u not in set(smoke_train) | set(smoke_ho)]
    order = smoke_train + smoke_ho + rest
    assert len(order) == len(set(order)) == len(train) + len(f0) + len(gold_on)
    for name, obj in (("train_ids.json", train), ("fold0_ondisk.json", f0), ("fold0_all.json", f0_all), ("gold58.json", gold_on),
                      ("smoke_train.json", smoke_train), ("smoke_ho.json", smoke_ho), ("precompute_order.json", order)):
        SC.atomic_json(obj, os.path.join(SC.DATA_DIR, name))
    print("wrote", SC.DATA_DIR, "| precompute studies", len(order))


if __name__ == "__main__":
    main()

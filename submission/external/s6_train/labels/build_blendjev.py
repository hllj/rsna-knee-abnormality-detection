#!/usr/bin/env python3
"""Rebuild labels_v1_blendjev.parquet: the cell-wise mean of labels_v1_blend (nartaa/rsna-knee-hpo-assets) and
riadmohamed42's JEV HYBRID_labels_scores.csv (riadmohamed42/jev-knee-labels, CC0); where the JEV cell is NaN the blend
value is kept. Row set and order = labels_v1_blend (4,349 studies). Float32 output; compare values, not file bytes.

  python build_blendjev.py labels_v1_blend.parquet HYBRID_labels_scores.csv out.parquet [labels_v1_blendjev.parquet]
"""
import sys
import numpy as np, pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
blend_p, jev_p, out_p = sys.argv[1:4]
B = pd.read_parquet(blend_p)
J = pd.read_csv(jev_p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
b = B[LAB].to_numpy(np.float64)
j = J.reindex(B["StudyInstanceUID"])[LAB].to_numpy(np.float64)
out = B[["StudyInstanceUID"]].copy()
out[LAB] = np.where(np.isnan(j), b, (b + j) / 2).astype(np.float32)
out.to_parquet(out_p, index=False)
print("rows %d, JEV NaN cells kept from blend %d" % (len(out), int(np.isnan(j).sum())))
if len(sys.argv) > 4:
    ref = pd.read_parquet(sys.argv[4]).set_index("StudyInstanceUID").loc[out["StudyInstanceUID"], LAB]
    print("max |rebuild - reference| = %.3g" % np.abs(out[LAB].to_numpy(np.float64) - ref.to_numpy(np.float64)).max())

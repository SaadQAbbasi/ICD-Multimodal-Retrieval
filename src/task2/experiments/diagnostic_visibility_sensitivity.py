"""Sensitivity check for the "chest-visible" share of positive labels.

diagnostic_code_groups.py uses one hand-picked list of ICD-9 categories that a
chest X-ray (or its radiology report) can show. This script tags every
diagnosis code with a tier so the share can be quoted as a range:

  narrow  = the original CHEST_VISIBLE list (lung/pleura/heart-failure/etc.)
  broad   = narrow + other conditions whose sign a CXR or its report can show
            (devices and post-surgical status, cardiomyopathy, valve disease,
            aortic disease, hiatal hernia, thoracic bone findings, ...)
  device  = codes inferable mainly from a visible line/catheter (dialysis)
  other   = everything else (no direct radiological sign)

Per tier: #codes, share of test positives, baseline / model micro AP, lift.
Also: the "other" tier broken down by chapter, and its top codes.

    python -m experiments.diagnostic_visibility_sensitivity \
        outputs/experiment05_ft_supcon_w0.001_both_s42
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.diagnostic_code_groups import CATALOG, CHAPTERS, CHEST_VISIBLE, micro_ap, three_digit
from experiments.experiment01 import ExperimentConfig, load_admission_records, split_records

ROOT = Path(__file__).resolve().parents[1]
D_ICD = os.environ.get("D_ICD_DIAGNOSES", "/datasets/MIMIC-IV/files/mimiciv/3.1/hosp/d_icd_diagnoses.csv.gz")  # code descriptions, listing only

# Extra prefixes (ICD-9 code without the dot) for the BROAD tier.
BROAD_EXTRA = {
    "011": "pulmonary tuberculosis", "1363": "pneumocystis pneumonia",
    "163": "pleural malignancy", "164": "thymus/mediastinal malignancy", "1961": "intrathoracic lymph node mets",
    "1985": "bone metastases", "2120": "benign neoplasm resp.", "2123": "benign neoplasm resp.",
    "4029": "hypertensive heart disease", "4021": "hypertensive heart disease", "4020": "hypertensive heart disease",
    "4040": "hypertensive heart+kidney disease", "4041": "hypertensive heart+kidney disease", "4049": "hypertensive heart+kidney disease",
    "416": "pulmonary hypertension", "4150": "acute cor pulmonale", "423": "pericardial disease",
    "424": "valve disorders", "425": "cardiomyopathy", "4400": "aortic atherosclerosis",
    "441": "aortic aneurysm/dissection", "493": "asthma", "494": "bronchiectasis",
    **{str(c): "pneumoconiosis/other lung" for c in range(495, 507)},
    "508": "resp. conditions due to other agents", "510": "empyema", "513": "lung abscess",
    "517": "lung involvement in other disease", "519": "other respiratory (mediastinum/diaphragm/trachea)",
    "5533": "hiatal hernia", "5503": "hiatal hernia", "7330": "osteoporosis",
    "73313": "vertebral pathological fracture", "737": "kyphosis/scoliosis",
    "8052": "thoracic vertebral fracture", "8053": "thoracic vertebral fracture", "806": "spine fracture with cord injury",
    "810": "clavicle fracture", "811": "scapula fracture", "861": "heart/lung injury", "862": "intrathoracic injury",
    "9973": "respiratory complication of procedure", "9960": "cardiac device complication",
    "V4501": "pacemaker in situ", "V4502": "defibrillator in situ", "V4581": "CABG status (sternotomy wires)",
    "V433": "heart valve replaced", "V440": "tracheostomy status", "V461": "ventilator dependence",
    "V426": "lung transplant", "V421": "heart transplant", "V5332": "defibrillator fitting", "V5331": "pacemaker fitting",
}
DEVICE_ONLY = {  # inferable mainly from a visible dialysis catheter
    "5856": "end stage renal disease", "V451": "dialysis status", "V560": "dialysis encounter", "99673": "dialysis device complication",
}


def code_of(meta):
    return meta["path_root_to_leaf"][-1].replace(".", "")


def match(code, table):
    return any(code.startswith(p) for p in table)


def chapter_of(meta):
    root = meta["path_root_to_leaf"][0]
    if root.startswith("V"):
        return "V codes (supplementary)"
    if root.startswith("E"):
        return "E codes (external causes)"
    start = int(root.split("-")[0].split(".")[0])
    return next(n for a, b, n in CHAPTERS if a <= start <= b)


def main():
    run = Path(sys.argv[1]).resolve()
    d = np.load(run / "test_predictions.npz")
    targets = d["targets"].astype(np.float32)
    fused = d["fused"].astype(np.float32)
    keys = json.loads((run / "labels.json").read_text())
    catalog = json.loads(CATALOG.read_text())
    cfg = ExperimentConfig(task2_root=str(ROOT)).resolve()
    train = split_records(load_admission_records(cfg.dataset_dir, cfg.mimic_root).records, "train")
    freq = Counter(k for r in train for k in set(r.target_keys))
    baseline = np.broadcast_to(np.array([freq[k] / len(train) for k in keys], dtype=np.float32), targets.shape)
    with gzip.open(D_ICD, "rt") as f:
        desc = {r["icd_code"]: r["long_title"] for r in csv.DictReader(f) if r["icd_version"] == "9"}

    tier, chapter = {}, {}
    for i, k in enumerate(keys):
        meta = catalog[k]
        if meta["code_system"] == "PCS":
            continue
        code = code_of(meta)
        chapter[i] = chapter_of(meta)
        if three_digit(meta["path_root_to_leaf"]) in CHEST_VISIBLE:
            tier[i] = "narrow"
        elif match(code, BROAD_EXTRA):
            tier[i] = "broad-extra"
        elif match(code, DEVICE_ONLY):
            tier[i] = "device"
        else:
            tier[i] = "other"

    pos = targets.sum(0)
    total = pos.sum()

    def stats(idx):
        idx = np.asarray(sorted(idx))
        b = micro_ap(targets[:, idx], baseline[:, idx])
        m = micro_ap(targets[:, idx], fused[:, idx])
        return len(idx), float(100 * pos[idx].sum() / total), b, m, m / b if b > 0 else float("nan")

    by_tier = defaultdict(list)
    for i, t in tier.items():
        by_tier[t].append(i)
    rows = [
        ("narrow (original list)", by_tier["narrow"]),
        ("broad extra codes only", by_tier["broad-extra"]),
        ("BROAD = narrow + extra", by_tier["narrow"] + by_tier["broad-extra"]),
        ("dialysis-catheter related", by_tier["device"]),
        ("no direct radiological sign", by_tier["other"]),
        ("all diagnoses", list(tier)),
    ]
    out = [f"Run: {run.name}. Share = fraction of all test positive (admission, code) pairs, procedures included in the denominator.",
           "", "| tier | codes | share of test positives | baseline AP | model fused AP | lift |", "|---|---|---|---|---|---|"]
    result = {"run": str(run), "tiers": {}}
    for name, idx in rows:
        n, s, b, m, l = stats(idx)
        out.append(f"| {name} | {n} | {s:.1f}% | {b:.3f} | {m:.3f} | {l:.2f}× |")
        result["tiers"][name] = {"codes": n, "pos_share": s, "baseline_ap": b, "fused_ap": m, "lift": l}

    out += ["", "No-direct-sign tier by chapter:", "", "| chapter | codes | share of test positives | baseline AP | model fused AP | lift |",
            "|---|---|---|---|---|---|"]
    by_ch = defaultdict(list)
    for i in by_tier["other"]:
        by_ch[chapter[i]].append(i)
    result["other_by_chapter"] = {}
    for ch, idx in sorted(by_ch.items(), key=lambda kv: -pos[kv[1]].sum()):
        n, s, b, m, l = stats(idx)
        out.append(f"| {ch} | {n} | {s:.1f}% | {b:.3f} | {m:.3f} | {l:.2f}× |")
        result["other_by_chapter"][ch] = {"codes": n, "pos_share": s, "baseline_ap": b, "fused_ap": m, "lift": l}

    for t, title in (("broad-extra", "Broad extra codes"), ("other", "No-direct-sign codes")):
        out += ["", f"Top {title} by share of positives:", ""]
        for i in sorted(by_tier[t], key=lambda i: -pos[i])[:25]:
            c = code_of(catalog[keys[i]])
            out.append(f"- {100 * pos[i] / total:.2f}%  {c}  {desc.get(c, '(category)')}")

    text = "\n".join(out) + "\n"
    print(text)
    (ROOT / "outputs/diagnostic_visibility_sensitivity_round5_s42.md").write_text(text)
    (ROOT / "outputs/diagnostic_visibility_sensitivity_round5_s42.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Diagnostic: chest-visible ICD categories vs what the CXR reports actually say.

For each chest finding (ICD-9 category <-> CheXpert report label), on the test
admissions:

  agreement     P(report+ | ICD+): coded condition is described in a report of the admission
                P(ICD+ | report+): described finding is coded for the admission
  granularity   model AP on the exact leaf codes vs on the whole category
                (category score = max leaf probability; label = any leaf code)
  reader        AP of the CheXpert label itself (fraction of the admission's
                studies positive) used as the "prediction" of the ICD category
                = what a perfect reader of the reports would get (rough ceiling)
  model sees    AUROC/AP of the model's category score against the CheXpert
                finding (does the model detect the finding itself?)

CheXpert labels: /datasets/MIMIC-CXR/mimic-cxr-2.0.0-chexpert.csv.gz (1 = positive;
-1 uncertain counted negative; also reported with uncertain = positive).

    python -m experiments.diagnostic_chest_findings outputs/experiment05_ft_supcon_w0.001_both_s42
"""
from __future__ import annotations

import gzip
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.experiment01 import ExperimentConfig, load_admission_records

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "outputs/experiment03_supcon_w0.001_e120_cos_s42/target_catalog.json"
CHEXPERT = Path(ExperimentConfig.mimic_root) / "mimic-cxr-2.0.0-chexpert.csv.gz"  # MIMIC_ROOT env

# name -> (ICD node prefixes (ICD-9 nodes or PCS code prefixes), CheXpert columns)
FINDINGS = {
    "Pleural effusion": (["511"], ["Pleural Effusion"]),
    "Pneumothorax": (["512", "860"], ["Pneumothorax"]),
    "Pneumonia": ([str(c) for c in range(480, 487)] + ["507"], ["Pneumonia"]),
    "Pneumonia (report: pneumonia or consolidation)": ([str(c) for c in range(480, 487)] + ["507"], ["Pneumonia", "Consolidation"]),
    "Heart failure (428) vs edema": (["428"], ["Edema"]),
    "Heart failure (428) vs edema or cardiomegaly": (["428"], ["Edema", "Cardiomegaly"]),
    "Acute pulmonary edema (518.4, 514)": (["518.4", "514"], ["Edema"]),
    "Atelectasis (518.0)": (["518.0"], ["Atelectasis"]),
    "Cardiomegaly (429.3)": (["429.3"], ["Cardiomegaly"]),
    "Rib/sternum fracture (807)": (["807"], ["Fracture"]),
    "Lung cancer / mass (162, 197.0, 793.1)": (["162", "197.0", "793.1"], ["Lung Lesion"]),
    "Mechanical ventilation / intubation (PCS)": (["5A19", "0BH17EZ", "0BH18EZ"], ["Support Devices"]),
    "Central venous line (PCS 02HV33Z)": (["02HV33Z"], ["Support Devices"]),
}


def matches(meta, prefixes):
    if meta["code_system"] == "PCS":
        code = meta["path_root_to_leaf"][-1]
        return any(code.startswith(p) for p in prefixes)
    return any(node == p or node.startswith(p + ".") or (("." in p) and node.startswith(p))
               for node in meta["path_root_to_leaf"] for p in prefixes)


def safe(fn, y, s):
    return float(fn(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main():
    run = Path(sys.argv[1]).resolve()
    d = np.load(run / "test_predictions.npz")
    hadm = d["hadm_ids"]
    label_keys = json.loads((run / "labels.json").read_text())
    catalog = json.loads(CATALOG.read_text())
    cfg = ExperimentConfig(task2_root=str(ROOT)).resolve()
    records = {r.hadm_id: r for r in load_admission_records(cfg.dataset_dir, cfg.mimic_root).records}
    recs = [records[int(h)] for h in hadm]

    chex = {}
    with gzip.open(CHEXPERT, "rt") as fh:
        for row in csv.DictReader(fh):
            chex[int(row["study_id"])] = row

    rows = []
    for name, (prefixes, columns) in FINDINGS.items():
        leaf = np.array([i for i, k in enumerate(label_keys) if matches(catalog[k], prefixes)])
        cat_keys = {k for k in catalog if matches(catalog[k], prefixes)}
        y = np.array([any(k in cat_keys for k in r.target_keys) for r in recs], dtype=np.float32)

        def report_score(r, uncertain_positive):
            vals = []
            for st in r.studies:
                row = chex.get(st.study_id)
                if row is None:
                    continue
                v = [row[c] for c in columns]
                pos = any(x == "1.0" or (uncertain_positive and x == "-1.0") for x in v)
                vals.append(float(pos))
            return float(np.mean(vals)) if vals else 0.0

        rep = np.array([report_score(r, False) for r in recs], dtype=np.float32)
        rep_u = np.array([report_score(r, True) for r in recs], dtype=np.float32)
        c, c_u = (rep > 0).astype(np.float32), (rep_u > 0).astype(np.float32)
        row = {"finding": name, "leaf_codes_in_vocab": int(len(leaf)), "test_admissions": len(recs),
               "icd_pos": int(y.sum()), "report_pos": int(c.sum()), "report_pos_with_uncertain": int(c_u.sum()),
               "p_report_given_icd": float(c[y == 1].mean()) if y.sum() else float("nan"),
               "p_icd_given_report": float(y[c == 1].mean()) if c.sum() else float("nan"),
               "baseline_ap": float(y.mean())}
        if len(leaf):
            t_leaf = d["targets"][:, leaf].astype(np.float32)
            for m in ("fused", "image", "text"):
                p_leaf = d[m][:, leaf].astype(np.float32)
                s = p_leaf.max(1)
                if m == "fused":
                    row["model_ap_exact_codes"] = float(average_precision_score(t_leaf.ravel(), p_leaf.ravel())) if t_leaf.any() else float("nan")
                row[f"model_{m}_ap_category"] = safe(average_precision_score, y, s)
                row[f"model_{m}_auroc_category"] = safe(roc_auc_score, y, s)
                row[f"model_{m}_auroc_vs_report"] = safe(roc_auc_score, c, s)
            s = d["fused"][:, leaf].astype(np.float32).max(1)
            row["model_fused_ap_vs_report"] = safe(average_precision_score, c, s)
            row["reader_plus_model_ap"] = safe(average_precision_score, y, rep + 1e-3 * s)
        row["reader_ap"] = safe(average_precision_score, y, rep)
        row["reader_ap_uncertain_pos"] = safe(average_precision_score, y, rep_u)
        row["reader_auroc"] = safe(roc_auc_score, y, rep)
        rows.append(row)

    f = lambda v: "–" if v != v else f"{v:.2f}"
    lines = [
        "| finding (ICD category ↔ report label) | ICD+ / report+ (of %d) | P(report+ given ICD+) | P(ICD+ given report+) "
        "| baseline AP | model AP exact codes | model AP category | model AUROC category | reader AP (report label → ICD) "
        "| model AUROC vs report label (fused / image) |" % len(recs),
        "|" + "---|" * 10,
    ]
    for r in rows:
        lines.append(
            f"| {r['finding']} | {r['icd_pos']} / {r['report_pos']} | {f(r['p_report_given_icd'])} | {f(r['p_icd_given_report'])} "
            f"| {f(r['baseline_ap'])} | {f(r.get('model_ap_exact_codes', float('nan')))} | {f(r.get('model_fused_ap_category', float('nan')))} "
            f"| {f(r.get('model_fused_auroc_category', float('nan')))} | {f(r['reader_ap'])} "
            f"| {f(r.get('model_fused_auroc_vs_report', float('nan')))} / {f(r.get('model_image_auroc_vs_report', float('nan')))} |")
    table = "\n".join(lines)
    print(table)
    out = ROOT / "outputs/diagnostic_chest_findings_round5_s42"
    out.with_suffix(".json").write_text(json.dumps({"run": str(run), "rows": rows}, indent=2))
    out.with_suffix(".md").write_text(table + "\n")
    print("wrote", out.with_suffix(".md"))


if __name__ == "__main__":
    main()

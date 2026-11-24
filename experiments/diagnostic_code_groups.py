"""Diagnostic: where does test AP come from? Split codes into groups and compare
the model with the training-frequency baseline (same score for every admission
= train prevalence of the code).

Groups: code type (diagnosis ICD-9-CM / procedure ICD-10-PCS), diagnosis
chapter, a hand-picked "visible on chest X-ray / in a CXR report" diagnosis set,
procedure body system, and training-frequency buckets.

Per group: #codes, share of test positive labels, micro AP (fused / image /
text / baseline), lift = AP / baseline AP, macro AP over codes with >= 5 test
positives, and the fine-tuning delta (second run minus first).

    python -m experiments.diagnostic_code_groups \
        outputs/experiment05_ft_supcon_w0.001_frozen_s42 outputs/experiment05_ft_supcon_w0.001_both_s42
"""
from __future__ import annotations

import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from experiments.experiment01 import ExperimentConfig, load_admission_records, split_records

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "outputs/experiment03_supcon_w0.001_e120_cos_s42/target_catalog.json"

CHAPTERS = [  # ICD-9-CM chapters by numeric start
    (1, 139, "infectious"), (140, 239, "neoplasms"), (240, 279, "endocrine/metabolic"),
    (280, 289, "blood"), (290, 319, "mental"), (320, 389, "nervous/sense"),
    (390, 459, "circulatory"), (460, 519, "respiratory"), (520, 579, "digestive"),
    (580, 629, "genitourinary"), (630, 679, "pregnancy"), (680, 709, "skin"),
    (710, 739, "musculoskeletal"), (740, 759, "congenital"), (760, 779, "perinatal"),
    (780, 799, "symptoms/ill-defined"), (800, 999, "injury/poisoning"),
]
CHEST_VISIBLE = {  # ICD-9 3-digit categories a CXR or its report can plausibly show
    **{str(c): "pneumonia" for c in range(480, 487)},
    "511": "pleural effusion", "512": "pneumothorax", "860": "pneumothorax",
    "514": "pulmonary congestion", "518": "lung collapse/resp. failure/other lung",
    "428": "heart failure", "429": "cardiomegaly/other heart",
    "162": "lung cancer", "197": "secondary resp. neoplasm",
    "491": "COPD/emphysema", "492": "COPD/emphysema", "496": "COPD/emphysema",
    "515": "interstitial lung disease", "516": "interstitial lung disease",
    "807": "rib/sternum fracture", "793": "abnormal imaging finding", "507": "aspiration pneumonitis",
}


def three_digit(path):
    for node in path:
        if "-" not in node and "." not in node:
            return node
    return path[-1].split(".")[0]


def groups_for(label_keys, catalog, train_freq):
    g: "OrderedDict[str, list[int]]" = OrderedDict()

    def add(name, i):
        g.setdefault(name, []).append(i)

    for i, key in enumerate(label_keys):
        meta = catalog[key]
        add("ALL", i)
        if meta["code_system"] == "PCS":
            add("type: procedures (ICD-10-PCS)", i)
            code = meta["path_root_to_leaf"][-1]
            if code.startswith(("0B", "5A19", "0W99", "0W9B", "02HV")):
                add("proc: airway/ventilation/chest tube/central line", i)
            elif code.startswith("02"):
                add("proc: heart & great vessels", i)
            elif code.startswith("B"):
                add("proc: imaging", i)
            else:
                add("proc: other", i)
        else:
            add("type: diagnoses (ICD-9-CM)", i)
            path = meta["path_root_to_leaf"]
            root = path[0]
            if root.startswith("V"):
                chapter = "V codes (supplementary)"
            elif root.startswith("E"):
                chapter = "E codes (external causes)"
            else:
                start = int(root.split("-")[0].split(".")[0])
                chapter = next(n for a, b, n in CHAPTERS if a <= start <= b)
            add(f"dx chapter: {chapter}", i)
            if three_digit(path) in CHEST_VISIBLE:
                add("dx: CHEST-VISIBLE set", i)
            else:
                add("dx: all other diagnoses", i)
        f = train_freq[key]
        bucket = "1" if f == 1 else "2-9" if f < 10 else "10-99" if f < 100 else "100-999" if f < 1000 else ">=1000"
        add(f"train freq: {bucket}", i)
    return g


def micro_ap(t, p):
    return float(average_precision_score(t.ravel(), p.ravel())) if t.any() else float("nan")


def macro_ap(t, p, min_pos=5):
    keep = t.sum(0) >= min_pos
    if not keep.any():
        return float("nan"), 0
    return float(np.mean([average_precision_score(t[:, j], p[:, j]) for j in np.where(keep)[0]])), int(keep.sum())


def main():
    runs = [Path(a).resolve() for a in sys.argv[1:]]
    data = [np.load(r / "test_predictions.npz") for r in runs]
    targets = data[0]["targets"].astype(np.float32)
    for d in data[1:]:
        assert (d["hadm_ids"] == data[0]["hadm_ids"]).all()
    label_keys = json.loads((runs[0] / "labels.json").read_text())
    catalog = json.loads(CATALOG.read_text())
    cfg = ExperimentConfig(task2_root=str(ROOT)).resolve()
    train = split_records(load_admission_records(cfg.dataset_dir, cfg.mimic_root).records, "train")
    train_freq = Counter(k for r in train for k in set(r.target_keys))
    prevalence = np.array([train_freq[k] / len(train) for k in label_keys], dtype=np.float32)
    baseline = np.broadcast_to(prevalence, targets.shape)

    groups = groups_for(label_keys, catalog, train_freq)
    total_pos = targets.sum()
    rows = []
    names = [r.name.replace("experiment05_ft_supcon_w0.001_", "").replace("_s42", "") for r in runs]
    for name, idx in groups.items():
        idx = np.asarray(idx)
        t = targets[:, idx]
        row = {"group": name, "codes": len(idx), "codes_with_test_pos": int((t.sum(0) > 0).sum()),
               "pos_share": float(t.sum() / total_pos), "baseline_ap": micro_ap(t, baseline[:, idx])}
        for run_name, d in zip(names, data):
            for m in ("fused", "image", "text"):
                row[f"{run_name}_{m}_ap"] = micro_ap(t, d[m][:, idx].astype(np.float32))
            row[f"{run_name}_macro_ap"], row["macro_codes"] = macro_ap(t, d["fused"][:, idx].astype(np.float32))
        row["baseline_macro_ap"], _ = macro_ap(t, baseline[:, idx])
        rows.append(row)

    last = names[-1]
    hdr = (f"| group | codes | share of test positives | baseline AP | {last} fused AP | lift | image AP | text AP "
           f"| macro AP (≥5 pos) vs baseline | fine-tune Δ ({names[-1]}−{names[0]}) |")
    lines = [hdr, "|" + "---|" * 10]
    for r in rows:
        lift = r[f"{last}_fused_ap"] / r["baseline_ap"] if r["baseline_ap"] > 0 else float("nan")
        delta = r[f"{last}_fused_ap"] - r[f"{names[0]}_fused_ap"]
        lines.append(
            f"| {r['group']} | {r['codes']} | {100*r['pos_share']:.1f}% | {r['baseline_ap']:.3f} | {r[f'{last}_fused_ap']:.3f} "
            f"| {lift:.2f}× | {r[f'{last}_image_ap']:.3f} | {r[f'{last}_text_ap']:.3f} "
            f"| {r[f'{last}_macro_ap']:.3f} vs {r['baseline_macro_ap']:.3f} (n={r['macro_codes']}) | {delta:+.4f} |")
    table = "\n".join(lines)
    print(table)
    out = ROOT / "outputs/diagnostic_code_groups_round5_s42.json"
    out.write_text(json.dumps({"runs": [str(r) for r in runs], "rows": rows}, indent=2))
    (ROOT / "outputs/diagnostic_code_groups_round5_s42.md").write_text(table + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()

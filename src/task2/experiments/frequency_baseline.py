"""Frequency baseline: predict every code with its training-set prevalence.

Every test admission gets the same score vector (the fraction of training
admissions that carry each code), so micro AP = what "always guess the common
codes" achieves. The script evaluates it on exact codes and on each ICD
hierarchy level and writes a side-by-side table with a trained run.

    python -m experiments.frequency_baseline [outputs/<run_dir>]
    (default run dir: outputs/experiment03_supcon_w0.001_e120_cos_s42)
-> <run_dir>/frequency_baseline.json, <run_dir>/frequency_baseline_comparison.csv
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from experiments.experiment01 import load_admission_records, split_records, multilabel_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'outputs/experiment03_supcon_w0.001_e120_cos_s42'


def constant_metrics(y, frequencies, threshold, top_k):
    # For constant predictions, each label's AP is prevalence and AUROC is 0.5.
    n = len(y)
    counts = y.sum(axis=0)
    predicted = frequencies >= threshold
    tp = counts * predicted
    fp = (n-counts) * predicted
    fn = counts * (~predicted)
    denom = 2*tp+fp+fn
    valid = (counts>0)&(counts<n)
    scores = np.broadcast_to(frequencies,y.shape).copy()
    k = min(top_k,len(frequencies))
    top = np.argpartition(-frequencies,kth=k-1)[:k]
    hits = y[:,top].sum(axis=1)
    return {
        'micro_f1':float(2*tp.sum()/max(denom.sum(),1)),
        'macro_f1':float(np.divide(2*tp,denom,out=np.zeros_like(tp),where=denom>0).mean()),
        'micro_ap':float(average_precision_score(y.ravel(),scores.ravel())),
        'evaluated_label_fraction':float(valid.mean()),
        'macro_ap_valid':float((counts[valid]/n).mean()) if valid.any() else float('nan'),
        'macro_auroc_valid':0.5 if valid.any() else float('nan'),
        f'precision_at_{k}':float((hits/k).mean()),
        f'recall_at_{k}':float((hits/np.maximum(y.sum(axis=1),1)).mean()),
    }

def run():
    manifest = json.loads((OUT / 'manifest.json').read_text())
    saved = json.loads((OUT / 'results.json').read_text())
    labels = json.loads((OUT / 'labels.json').read_text())
    cfg = manifest['config']
    loaded = load_admission_records(cfg['dataset_dir'], cfg['mimic_root'])
    catalog = loaded.target_catalog
    vocab = set(labels)
    cohorts = {s: [r for r in split_records(loaded.records, s) if vocab.intersection(r.target_keys)]
               for s in ['train', 'validation', 'test']}
    cohorts['test_complete_images'] = [r for r in cohorts['test'] if r.all_referenced_images_available]
    assert len(cohorts['test']) == saved['test']['sample_count']
    assert len(cohorts['test_complete_images']) == saved['test_complete_images']['sample_count']
    report = {'method': 'Per-level admission prevalence, counted once per class per training admission',
              'training_admissions': len(cohorts['train']),
              'f1_note': 'Fixed threshold from model config for direct comparison; no threshold tuning',
              'tie_note': 'Top-K uses the same numpy argpartition implementation as the model evaluator',
              'evaluations': {}}
    comparison = []
    settings = [('exact_all_targets', list(labels), lambda r: set(r.target_keys))]
    cm_labels = [k for k in labels if catalog[k]['code_system'] == 'CM']
    depth = max(len(catalog[k]['path_root_to_leaf']) for k in cm_labels)
    def prefix(key, level):
        m = catalog[key]
        return (m['mapped_version'], m['code_system'], *m['path_root_to_leaf'][:level])
    for level in range(1, depth + 1):
        classes = sorted({prefix(k, level) for k in cm_labels}, key=lambda v: tuple(map(str, v)))
        settings.append((f'level_{level}', classes,
                         lambda r, level=level: {prefix(k, level) for k in r.target_keys if catalog[k]['code_system'] == 'CM'}))
    for name, classes, target_set in settings:
        indices = {k: i for i, k in enumerate(classes)}
        def targets(records):
            y = np.zeros((len(records), len(classes)), dtype=np.float32)
            for i, r in enumerate(records):
                for k in target_set(r):
                    if k in indices: y[i, indices[k]] = 1
            return y
        train_y = targets(cohorts['train'])
        frequencies = train_y.mean(axis=0)
        top = np.argsort(-frequencies, kind='stable')[:10]
        report['evaluations'][name] = {'class_count': len(classes),
            'most_common': [{'class': classes[i], 'frequency': float(frequencies[i])} for i in top]}
        for cohort in ['validation', 'test', 'test_complete_images']:
            y = targets(cohorts[cohort])
            scores = np.broadcast_to(frequencies, y.shape).copy()
            print(f'Evaluating {name}: {cohort}', flush=True)
            metrics = constant_metrics(y, frequencies, cfg['threshold'], cfg['top_k'])
            report['evaluations'][name][cohort] = {'sample_count':len(y), **metrics}
            if cohort == 'validation': continue
            model_metrics = saved[cohort]['modalities'] if name == 'exact_all_targets' else saved[cohort]['cm_hierarchy'][name]
            for method in ['frequency', 'image', 'text', 'fused']:
                values = metrics if method == 'frequency' else model_metrics[method]
                comparison.append({'cohort':cohort,'evaluation':name,'method':method,
                    **{k:values[k] for k in ['micro_ap','micro_f1','macro_ap_valid','precision_at_5','recall_at_5']}})
    (OUT / 'frequency_baseline.json').write_text(json.dumps(report, indent=2))
    table = pd.DataFrame(comparison)
    table.to_csv(OUT / 'frequency_baseline_comparison.csv', index=False)
    print(table[table.cohort == 'test'].to_string(index=False))
    print('Saved:', OUT / 'frequency_baseline.json')
    print('Saved:', OUT / 'frequency_baseline_comparison.csv')

if __name__ == '__main__':
    run()

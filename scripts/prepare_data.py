"""Reproduce Data v1 and audits. Only existing pandas/numpy dependencies are used."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lab3.data import load_hourly, holdout_start

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n',
                    encoding='utf-8', newline='\n')

def prepare(raw_path, output_dir, config, source_manifest):
    raw_path, output_dir = Path(raw_path), Path(output_dir)
    raw_sha = sha(raw_path)
    if raw_sha != source_manifest['sha256']:
        raise ValueError('Raw source hash differs from original manifest.')
    frame, quality = load_hourly(raw_path)
    cutoff = holdout_start(frame.index, config['train_ratio'])
    if str(cutoff) != config['test_start']:
        raise ValueError('Frozen test boundary changed.')
    raw = pd.read_csv(raw_path)
    raw['Datetime'] = pd.to_datetime(raw['Datetime'], errors='raise')
    duplicate_rows = raw.loc[raw['Datetime'].duplicated(keep=False)].sort_values(
        'Datetime', kind='stable')
    missing_hours = frame.index[~frame['observed']]
    train = frame.loc[frame.index < cutoff]
    test = frame.loc[frame.index >= cutoff]
    quality.update({
        'version': 'v1', 'prepared_by': 'assistant',
        'prepared_date': config['data_ready_date'], 'raw_sha256': raw_sha,
        'source_url': source_manifest['source'], 'unit': 'MW',
        'observed_targets': int(frame['observed'].sum()),
        'duplicate_audit_rows': len(duplicate_rows),
        'missing_target_policy': 'keep_NaN_and_observed_False_do_not_score',
        'history_policy': 'causal_ffill_only_inside_feature_builder',
        'columns': {'Datetime': 'nominal local hour, no timezone',
                    'y': 'AEP_MW after duplicate mean; missing values remain blank',
                    'observed': 'True if an original target was observed at this nominal hour'},
        'split': {'test_start': str(cutoff), 'train_ratio': config['train_ratio'],
                  'train_grid_rows': len(train), 'test_grid_rows': len(test),
                  'train_observed_targets': int(train['observed'].sum()),
                  'test_observed_targets': int(test['observed'].sum())},
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir/'AEP_hourly_clean.csv', index_label='Datetime',
                 date_format='%Y-%m-%d %H:%M:%S', lineterminator='\n')
    duplicate_rows.to_csv(output_dir/'duplicate_timestamps.csv', index=False,
                          date_format='%Y-%m-%d %H:%M:%S', lineterminator='\n')
    pd.DataFrame({'Datetime': missing_hours}).to_csv(output_dir/'missing_hours.csv', index=False,
                          date_format='%Y-%m-%d %H:%M:%S', lineterminator='\n')
    write_json(output_dir/'data_quality.json', quality)
    names = ['AEP_hourly_clean.csv', 'duplicate_timestamps.csv', 'missing_hours.csv', 'data_quality.json']
    write_json(output_dir/'manifest.json', {'version': 'v1', 'raw_sha256': raw_sha,
                'files': {name: sha(output_dir/name) for name in names}})
    return quality

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/interim')
    args = parser.parse_args()
    cfg = json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
    source = json.loads((ROOT/'data/raw/manifest.json').read_text(encoding='utf-8'))
    q = prepare(ROOT/cfg['dataset_path'], args.output_dir, cfg, source)
    print('DATA_V1_READY raw_rows=%d grid_rows=%d duplicates=%d missing=%d' % (
        q['raw_rows'], q['grid_rows'], q['duplicate_timestamp_rows'], q['missing_nominal_hours']))

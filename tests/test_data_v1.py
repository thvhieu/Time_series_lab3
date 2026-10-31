from pathlib import Path
import hashlib
import json
import sys
import tempfile
import unittest
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT))
from lab3.data import load_hourly, load_prepared, split_supervised
from lab3.features import make_supervised
from scripts.prepare_data import prepare

class PreparedDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT/'data/interim'
        cls.raw = ROOT/'data/raw/AEP_hourly.csv'
        cls.cfg = json.loads((ROOT/'configs/experiment.json').read_text(encoding='utf-8'))
        cls.source = json.loads((ROOT/'data/raw/manifest.json').read_text(encoding='utf-8'))
        cls.frame, cls.quality = load_prepared(cls.folder, cls.raw)

    def test_raw_hash_unchanged(self):
        self.assertEqual(hashlib.sha256(self.raw.read_bytes()).hexdigest(), self.source['sha256'])

    def test_quality_counts(self):
        q = self.quality
        self.assertEqual((q['raw_rows'], q['grid_rows'], q['duplicate_timestamp_rows'],
                          q['missing_nominal_hours'], q['observed_targets']),
                         (121273, 121296, 4, 27, 121269))

    def test_prepared_equals_original_loader(self):
        original, _ = load_hourly(self.raw)
        pd.testing.assert_frame_equal(original, self.frame)

    def test_missing_targets_are_not_imputed(self):
        self.assertEqual(int(self.frame.y.isna().sum()), 27)
        self.assertTrue(self.frame.y.notna().equals(self.frame.observed))

    def test_audit_files(self):
        missing = pd.read_csv(self.folder/'missing_hours.csv')
        duplicates = pd.read_csv(self.folder/'duplicate_timestamps.csv')
        self.assertEqual(len(missing), 27)
        self.assertEqual(len(duplicates), 8)
        self.assertTrue(duplicates.Datetime.duplicated(keep=False).all())

    def test_split_boundary_unchanged(self):
        cfg = self.cfg
        s = make_supervised(self.frame, cfg['default_p'], tuple(cfg['rolling_windows']), cfg['warmup_hours'])
        train, test = split_supervised(s, cfg['test_start'])
        self.assertLess(train.index[-1], test.index[0])
        self.assertEqual(self.quality['split']['test_start'], '2015-10-27 05:00:00')
        self.assertEqual(self.quality['split']['train_grid_rows'], 97036)
        self.assertEqual(self.quality['split']['test_grid_rows'], 24260)

    def test_reproducible_bytes(self):
        with tempfile.TemporaryDirectory(prefix='lab3-v1-test-') as tmp:
            folder = Path(tmp)
            prepare(self.raw, folder, self.cfg, self.source)
            for name in ('AEP_hourly_clean.csv', 'data_quality.json', 'duplicate_timestamps.csv',
                         'missing_hours.csv', 'manifest.json'):
                self.assertEqual((folder/name).read_bytes(), (self.folder/name).read_bytes(), name)

    def test_tampered_artifact_rejected(self):
        import shutil
        with tempfile.TemporaryDirectory(prefix='lab3-v1-tamper-') as tmp:
            folder = Path(tmp)
            for p in self.folder.iterdir():
                if p.is_file():
                    shutil.copy2(p, folder/p.name)
            with (folder/'missing_hours.csv').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_prepared(folder, self.raw)

    def test_no_member_delivery_on_29(self):
        team = json.loads((ROOT/'docs/team.json').read_text(encoding='utf-8'))
        for member in team:
            self.assertEqual(member['work_start_date'], '2026-10-30')
            self.assertGreaterEqual(member['first_handoff_date'], '2026-10-30')
        tv02 = next(m for m in team if m['id'] == 'TV02')
        self.assertEqual(tv02['focus'], 'eda_only')
        self.assertFalse(tv02['data_preparation_required'])

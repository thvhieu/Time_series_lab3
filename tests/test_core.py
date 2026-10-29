import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from lab3.data import load_hourly, holdout_start, split_supervised
from lab3.features import make_supervised, xy
from lab3.evaluation import regression_metrics
from lab3.modeling import make_model

def fixture(n=600):
    return pd.DataFrame({'y':np.arange(n,dtype=float)+10,'observed':True},
                        index=pd.date_range('2020-01-01',periods=n,freq='h'))

class CoreTests(unittest.TestCase):
    def test_real_data_quality(self):
        f,q=load_hourly(ROOT/'data/raw/AEP_hourly.csv')
        self.assertEqual(q['raw_rows'],121273)
        self.assertEqual(q['duplicate_timestamp_rows'],4)
        self.assertEqual(q['missing_nominal_hours'],27)
        self.assertEqual(int(f.y.isna().sum()),27)

    def test_real_split_frozen(self):
        f,_=load_hourly(ROOT/'data/raw/AEP_hourly.csv')
        c=json.loads((ROOT/'configs/experiment.json').read_text())
        self.assertEqual(str(holdout_start(f.index)),c['test_start'])

    def _load_csv(self,text):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'sample.csv';p.write_text(text,encoding='utf-8')
            return load_hourly(p)

    def test_duplicate_and_gap_policy(self):
        f,q=self._load_csv('Datetime,AEP_MW\n2020-01-01 02:00,4\n2020-01-01 00:00,2\n2020-01-01 00:00,6\n')
        self.assertEqual(f.y.iloc[0],4)
        self.assertTrue(pd.isna(f.y.iloc[1]))
        self.assertEqual(q['missing_nominal_hours'],1)

    def test_nonpositive_rejected(self):
        with self.assertRaises(ValueError):
            self._load_csv('Datetime,AEP_MW\n2020-01-01,0\n')

    def test_nonhour_rejected(self):
        with self.assertRaises(ValueError):
            self._load_csv('Datetime,AEP_MW\n2020-01-01 01:30,2\n')

    def test_lag_identity(self):
        f=fixture();s=make_supervised(f)
        self.assertEqual(s.lag_1.iloc[0],f.y.iloc[167])
        self.assertEqual(s.lag_24.iloc[0],f.y.iloc[144])
        self.assertEqual(s.y.iloc[0],f.y.iloc[168])

    def test_rolling_shift(self):
        f=fixture();s=make_supervised(f)
        self.assertAlmostEqual(s.roll_mean_24.iloc[0],f.y.iloc[144:168].mean())
        self.assertAlmostEqual(s.roll_std_24.iloc[0],f.y.iloc[144:168].std())

    def test_future_perturbation_does_not_change_past_features(self):
        f=fixture();a=make_supervised(f);g=f.copy();g.loc[g.index[300]:,'y']+=9999
        b=make_supervised(g)
        pd.testing.assert_frame_equal(a.drop(columns='y').loc[:f.index[300]],
                                      b.drop(columns='y').loc[:f.index[300]])

    def test_missing_target_not_scored(self):
        f=fixture();f.loc[f.index[200],'y']=np.nan;f.loc[f.index[200],'observed']=False
        s=make_supervised(f)
        self.assertNotIn(f.index[200],s.index)
        self.assertEqual(s.loc[f.index[201],'lag_1'],f.y.iloc[199])

    def test_common_warmup_for_p_candidates(self):
        f=fixture();base=make_supervised(f,p=6)
        for p in (12,24,48,168):
            self.assertTrue(base.index.equals(make_supervised(f,p=p).index))

    def test_invalid_lag_rejected(self):
        for p in (0,-1,1.5,True):
            with self.assertRaises(ValueError):make_supervised(fixture(),p=p)

    def test_warmup_rejected(self):
        with self.assertRaises(ValueError):make_supervised(fixture(),p=168,warmup=24)

    def test_irregular_index_rejected(self):
        with self.assertRaises(ValueError):make_supervised(fixture().drop(fixture().index[30]))

    def test_split_and_xy(self):
        f=fixture();s=make_supervised(f);a,b=split_supervised(s,holdout_start(f.index))
        self.assertLess(a.index[-1],b.index[0]);X,y=xy(a)
        self.assertNotIn('y',X.columns);self.assertTrue(X.index.equals(y.index))

    def test_invalid_split_rejected(self):
        with self.assertRaises(ValueError):holdout_start(fixture().index,1)
        with self.assertRaises(ValueError):split_supervised(make_supervised(fixture()),'1900-01-01')

    def test_metrics_perfect(self):
        y=fixture(10).y;m=regression_metrics(y,y)
        self.assertEqual((m['MAE'],m['RMSE'],m['MAPE']),(0,0,0))

    def test_metrics_exact(self):
        y=pd.Series([1.,3.],index=pd.date_range('2020',periods=2,freq='h'))
        m=regression_metrics(y,y+1)
        self.assertEqual(m['MAE'],1);self.assertEqual(m['RMSE'],1)
        self.assertAlmostEqual(m['MAPE'],(1+1/3)*50)

    def test_metrics_index_rejected(self):
        y=fixture(10).y
        with self.assertRaises(ValueError):regression_metrics(y,y.iloc[::-1])

    def test_metrics_nonfinite_rejected(self):
        y=fixture(10).y;p=y.copy();p.iloc[0]=np.nan
        with self.assertRaises(ValueError):regression_metrics(y,p)

    def test_mape_zero_count(self):
        y=pd.Series([0.,2.],index=pd.date_range('2020',periods=2,freq='h'))
        m=regression_metrics(y,y+1)
        self.assertEqual(m['mape_valid_n'],1);self.assertEqual(m['MAPE'],50)

    def test_factories_fit_synthetic_only(self):
        rng=np.random.default_rng(42);X=rng.normal(size=(100,4));y=X[:,0]*2+X[:,1]
        for name in ('linear','rf','xgboost'):
            kw={} if name=='linear' else {'n_estimators':5}
            model=make_model(name,**kw);model.fit(X[:80],y[:80]);p=model.predict(X[80:])
            self.assertEqual(p.shape,(20,));self.assertTrue(np.isfinite(p).all())

    def test_tss_stays_inside_train(self):
        from sklearn.model_selection import TimeSeriesSplit
        f=fixture();s=make_supervised(f);train,test=split_supervised(s,holdout_start(f.index))
        for a,b in TimeSeriesSplit(n_splits=3).split(train):
            self.assertLess(train.index[a[-1]],train.index[b[0]])
            self.assertLess(train.index[b[-1]],test.index[0])

import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_eda.evaluation.loso import choose, measure


class ReportTests(unittest.TestCase):
    def test_equal_subject_metric_convention(self):
        truth = [0,0,0,1,1,1]
        p = np.array([.1,.2,.6,.55,.3,.9])
        score = measure(pd.DataFrame(dict(y_true=truth,p0=1-p,p1=p)),2)
        self.assertAlmostEqual(score["brier"],brier_score_loss(truth,p))
        self.assertAlmostEqual(score["macro_f1"],f1_score(truth,p>.5,average="macro"))
        self.assertAlmostEqual(score["specificity"],2/3)

    def test_lightweight_uses_inner_scores_then_sensor_count_then_size(self):
        rows = pd.DataFrame(dict(name=["multi","large","small","weak"],score=[.81,.802,.801,.799],
                            sensors=[3,1,1,1],size=[80,100,90,1],outer_score=[0,0,0,1]))
        self.assertEqual(choose(rows),"multi")
        self.assertEqual(choose(rows,lightweight=True),"small")

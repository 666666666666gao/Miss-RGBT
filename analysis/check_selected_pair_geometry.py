"""CPU geometry fixtures; no image, model, tracking or benchmark evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis.diagnose_selected_pair_geometry import closed_pair, geometry, summarize


class GeometryChecks(unittest.TestCase):
    def test_identical_boxes(self):
        target = np.array([[10., 10., 20., 20.], [30., 40., 10., 30.]])
        result = summarize(geometry(target, target))
        self.assertEqual(result['valid_frames'], 2)
        self.assertEqual(result['center_error_px_median'], 0)
        self.assertEqual(result['mean_iou_pct'], 100)
        self.assertEqual(result['recentered_gain_pp'], 0)
        self.assertEqual(result['resized_gain_pp'], 0)

    def test_translation_preserves_size(self):
        target = np.array([[10., 10., 20., 20.]])
        prediction = np.array([[40., 50., 20., 20.]])
        before = prediction.copy()
        result = summarize(geometry(prediction, target))
        self.assertEqual(result['center_error_px_median'], 50)
        self.assertEqual(result['normalized_center_error_median'], 2.5)
        self.assertEqual(result['mean_iou_pct'], 0)
        self.assertEqual(result['recentered_mean_iou_pct'], 100)
        self.assertEqual(result['resized_mean_iou_pct'], 0)
        self.assertTrue(np.array_equal(prediction, before))

    def test_size_preserves_center(self):
        target = np.array([[10., 10., 20., 20.]])
        prediction = np.array([[0., 0., 40., 40.]])
        result = summarize(geometry(prediction, target))
        self.assertEqual(result['center_error_px_median'], 0)
        self.assertEqual(result['mean_iou_pct'], 25)
        self.assertEqual(result['recentered_mean_iou_pct'], 25)
        self.assertEqual(result['resized_mean_iou_pct'], 100)
        self.assertEqual(result['width_mape_pct'], 100)
        self.assertEqual(result['height_mape_pct'], 100)

    def test_full_closed_pair_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ('complete', 'miss')]
            for condition, root in zip(('complete', 'miss'), roots):
                root.mkdir()
                (root / 'run.json').write_text(json.dumps(dict(condition=condition, limit=None,
                    sequences=479, frames=337353, checkpoint_sha256='same-fixture-checkpoint')))
                (root / 'completed.json').write_text(json.dumps(dict(sequences=479, frames=337353)))
                (root / 'metrics.json').write_text('{}')
                (root / 'all_receipts.json').write_text('[]')
            self.assertEqual(closed_pair(*roots), 'same-fixture-checkpoint')
            (roots[1] / 'completed.json').unlink()
            with self.assertRaises(FileNotFoundError):
                closed_pair(*roots)

    def test_rejects_different_checkpoint_and_partial_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ('complete', 'miss')]
            for condition, root in zip(('complete', 'miss'), roots):
                root.mkdir()
                (root / 'run.json').write_text(json.dumps(dict(condition=condition, limit=None,
                    sequences=479, frames=337353, checkpoint_sha256=condition)))
                (root / 'completed.json').write_text(json.dumps(dict(sequences=479, frames=337353)))
                (root / 'metrics.json').write_text('{}')
                (root / 'all_receipts.json').write_text('[]')
            with self.assertRaises(AssertionError):
                closed_pair(*roots)
            run = json.loads((roots[1] / 'run.json').read_text())
            run.update(checkpoint_sha256='complete', limit=1)
            (roots[1] / 'run.json').write_text(json.dumps(run))
            with self.assertRaises(AssertionError):
                closed_pair(*roots)


if __name__ == '__main__':
    unittest.main()

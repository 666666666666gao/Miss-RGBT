"""CPU outer-gate fixtures; the image-cache loader is explicitly mocked."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from research.feedback_f.protocol_m1 import verify_decision
from research.protocol_r.common import ROOT, read, sha

sys.path.insert(0, str(ROOT / 'scripts'))
from scripts import diagnose_history_stream as diagnostic
from research.history_stream.diagnostic import different_sequence_donors, summarize


def write(path, value):
    path.write_text(json.dumps(value))


class DecisionChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / 'config.json'
        self.receipt_path = self.root / 'cache_receipt.json'
        self.weight = self.root / 'checkpoint.pth'
        self.cache_path = self.root / 'cache.pth'
        self.audit = self.root / 'audit.json'
        self.parent = self.root / 'parent.json'
        self.manifest = self.root / 'input_manifest.jsonl'
        write(self.config, {})
        self.weight.write_bytes(b'fixture checkpoint, not trained weights')
        self.cache_path.write_bytes(b'fixture metadata, not image tensors')
        write(self.audit, dict(status='passed', overlapping_frames=0))
        write(self.parent, dict(status='complete'))
        self.manifest.write_text('')
        self.cache = [dict(identity=dict(sequence=str(index % 16), sample=index))
                      for index in range(128)]
        self.receipt = dict(cache=str(self.cache_path),
            frame_exclusion_audit=str(self.audit),
            source_sha256={str(self.config): sha(self.config)},
            excluded_runs=[dict(completed=str(self.parent),
                completed_sha256=sha(self.parent), manifest=str(self.manifest),
                manifest_sha256=sha(self.manifest))])
        write(self.receipt_path, self.receipt)
        for name, value in [('CHECKPOINT', str(self.weight)),
                            ('CHECKPOINT_SHA', sha(self.weight))]:
            mock = patch.object(diagnostic, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        mock = patch.object(diagnostic, 'load_cache',
                            return_value=(self.receipt, self.cache))
        self.loader = mock.start()
        self.addCleanup(mock.stop)

    def report(self, name, layer, advance, previous=None):
        root = self.root / name
        root.mkdir()
        identities = [item['identity'] for item in self.cache]
        donors = different_sequence_donors(identities)
        rows = []
        for index, identity in enumerate(identities):
            for state in ('11', '01', '10'):
                row = dict(identity=identity, state=state,
                           wrong_sequence=identities[donors[index]]['sequence'])
                for mode in diagnostic.MODES:
                    iou = .5
                    if mode == 'true_hist' and state != '11':
                        iou += .1 if advance else -.1
                    row[mode] = dict(iou=iou, task_loss=2.)
                rows.append(row)
        effects, measured = summarize(rows, diagnostic.clustered)
        self.assertIs(measured, advance)
        value = dict(status='complete', injection_layer=layer, advance=advance,
            checkpoint=str(self.weight), checkpoint_sha256=sha(self.weight),
            cache_receipt=self.receipt, cache_receipt_sha256=sha(self.receipt_path),
            source_sha256={str(self.config): sha(self.config)},
            input_manifest=identities, rows=rows, effects=effects,
            m1_decision=diagnostic.decision(layer, advance),
            previous_diagnostic=previous, complete_bitwise_parity=True,
            native_zero_bitwise_parity=True, state_unchanged=True,
            gradients_absent=True)
        path = root / 'report.json'
        self.save_report(path, value)
        return path

    def save_report(self, path, value):
        write(path, value)
        write(path.parent / 'completed.json', dict(status='complete',
            report_sha256=sha(path), injection_layer=value['injection_layer'],
            advance=value['advance'], m1_decision=value['m1_decision']))

    def verify(self, path):
        return verify_decision(path, self.receipt_path, self.config)

    def previous(self, path):
        return diagnostic.check_retry(0, path, self.receipt_path, self.config)

    def test_missing_evidence_does_not_load_cache(self):
        result = self.verify(self.root / 'missing/report.json')
        self.assertEqual(result['stage'], 'WAITING_FOR_M1')
        self.assertFalse(result['ready'])
        self.assertFalse(result['f_training_authorized'])
        self.loader.assert_not_called()

    def test_all_four_decisions_without_f_authorization(self):
        positive = self.verify(self.report('block3_positive', 3, True))
        self.assertTrue(positive['ready'])
        self.assertTrue(positive['requires_ohcas_composition'])
        negative_path = self.report('block3_negative', 3, False)
        negative = self.verify(negative_path)
        self.assertFalse(negative['ready'])
        self.assertEqual(negative['stage'], 'M1_RETRY_REQUIRED')
        previous = self.previous(negative_path)
        for advance in (False, True):
            result = self.verify(self.report(
                'embedding_' + str(advance), 0, advance, previous))
            self.assertTrue(result['ready'])
            self.assertIs(result['requires_ohcas_composition'], advance)
            self.assertFalse(result['f_training_authorized'])
            self.assertIn(str(negative_path), result['parents_sha256'])
        for result in (positive, negative):
            self.assertFalse(result['f_training_authorized'])

    def test_changed_decision_rows_and_parity_are_rejected(self):
        path = self.report('positive', 3, True)
        original = read(path)
        alterations = [
            lambda r: r.update(m1_decision='closed_both_diagnostics_negative'),
            lambda r: r.update(complete_bitwise_parity=False),
            lambda r: r['rows'].pop(),
            lambda r: r['rows'][1].update(wrong_sequence='wrong'),
            lambda r: r['rows'][1]['true_hist'].update(iou=.95),
            lambda r: r.update(input_manifest=[]),
        ]
        for change in alterations:
            with self.subTest(change=alterations.index(change)):
                altered = copy.deepcopy(original)
                change(altered)
                self.save_report(path, altered)
                with self.assertRaises(AssertionError):
                    self.verify(path)

    def test_embedding_cannot_skip_negative_block3(self):
        path = self.report('embedding', 0, False)
        with self.assertRaises(AssertionError):
            self.verify(path)
        positive = self.report('positive', 3, True)
        report = read(path)
        report['previous_diagnostic'] = dict(report=str(positive))
        self.save_report(path, report)
        with self.assertRaises(AssertionError):
            self.verify(path)

    def test_cache_failure_is_not_a_negative_decision(self):
        path = self.report('negative', 3, False)
        self.loader.side_effect = AssertionError('historical exclusions unverified')
        with self.assertRaisesRegex(AssertionError, 'historical exclusions'):
            self.verify(path)

    def test_changed_source_checkpoint_and_terminal_hash_are_rejected(self):
        path = self.report('positive', 3, True)
        write(self.config, dict(changed=True))
        with self.assertRaises(AssertionError):
            self.verify(path)
        write(self.config, {})
        self.weight.write_bytes(b'changed weights')
        with self.assertRaises(AssertionError):
            self.verify(path)
        self.weight.write_bytes(b'fixture checkpoint, not trained weights')
        terminal = path.parent / 'completed.json'
        completed = read(terminal)
        completed['report_sha256'] = 'changed'
        write(terminal, completed)
        with self.assertRaises(AssertionError):
            self.verify(path)


if __name__ == '__main__':
    unittest.main()

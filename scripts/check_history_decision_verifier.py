"""Synthetic saved-report checks; no actual history diagnostic is executed."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import verify_history_stream as verifier
from research.history_stream.diagnostic import MODES, decision, summarize


def observations(advance):
    identities = [dict(sequence='fixture_%d' % (index % 16), search_frame=index)
                  for index in range(128)]
    donors = verifier.different_sequence_donors(identities)
    rows = []
    for index, identity in enumerate(identities):
        for state in ('11', '01', '10'):
            row = dict(identity=identity, state=state,
                       wrong_sequence=identities[donors[index]]['sequence'])
            for mode in MODES:
                iou = .5
                if state != '11' and mode == 'true_hist':
                    iou = .6 if advance else .4
                row[mode] = dict(iou=iou, task_loss=2.)
            rows.append(row)
    return identities, rows


def report_for(layer, advance):
    identities, rows = observations(advance)
    effects, measured = summarize(rows, verifier.clustered)
    assert measured is advance
    return dict(input_manifest=identities, rows=rows, effects=effects,
                injection_layer=layer, advance=advance,
                m1_decision=decision(layer, advance))


class DecisionVerifierTests(unittest.TestCase):
    def test_recomputes_all_four_decisions(self):
        for layer in (0, 3):
            for advance in (False, True):
                report = report_for(layer, advance)
                self.assertEqual(verifier.check_rows(report, report['input_manifest']),
                                 decision(layer, advance))

    def test_rejects_changed_pairs_or_reused_sequence(self):
        original = report_for(3, True)
        for change in ('raw_pair', 'donor', 'state', 'identity', 'missing_row',
                       'complete_parity', 'nonfinite', 'decision'):
            report = copy.deepcopy(original)
            if change == 'raw_pair':
                report['rows'][1]['true_hist']['iou'] = .4
            elif change == 'donor':
                report['rows'][0]['wrong_sequence'] = report['rows'][0]['identity']['sequence']
            elif change == 'state':
                report['rows'][1]['state'] = '10'
            elif change == 'identity':
                report['rows'][1]['identity'] = dict(sequence='unregistered')
            elif change == 'missing_row':
                report['rows'].pop()
            elif change == 'complete_parity':
                report['rows'][0]['true_hist']['iou'] = .6
            elif change == 'nonfinite':
                report['rows'][1]['true_hist']['iou'] = float('nan')
            else:
                report['m1_decision'] = 'closed_both_diagnostics_negative'
            with self.subTest(change=change), self.assertRaises(AssertionError):
                verifier.check_rows(report, report['input_manifest'])

    def test_saved_report_binds_sources_cache_checkpoint_and_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = Path(verifier.__file__).resolve().parents[1]
            config, weight, cache, receipt_path, audit = [
                root / name for name in
                ('config.json', 'weight.pth', 'cache.pth', 'receipt.json', 'audit.json')]
            for path in (config, weight, cache, audit):
                path.write_text('synthetic fixture')
            receipt = dict(cache=str(cache), frame_exclusion_audit=str(audit),
                           excluded_runs=[], source_sha256={})
            receipt_path.write_text(json.dumps(receipt))
            report = report_for(3, True)
            sources = {str(path): verifier.sha256(path) for path in (
                project / 'scripts/diagnose_history_stream.py',
                project / 'scripts/diagnose_absent_position.py',
                project / 'research/history_stream/diagnostic.py', config)}
            report.update(status='complete', checkpoint=str(weight),
                checkpoint_sha256=verifier.sha256(weight), source_sha256=sources,
                cache_receipt=receipt, cache_receipt_sha256=verifier.sha256(receipt_path),
                previous_diagnostic=None, complete_bitwise_parity=True,
                native_zero_bitwise_parity=True, state_unchanged=True,
                gradients_absent=True, seconds=1., peak_gpu_bytes=1)
            report_path = root / 'report.json'
            synthetic_cache = [dict(identity=item) for item in report['input_manifest']]

            def write(value=None, path=None):
                value = report if value is None else value
                path = report_path if path is None else path
                path.write_text(json.dumps(value))
                (path.parent / 'completed.json').write_text(json.dumps(dict(
                    status='complete', injection_layer=value['injection_layer'],
                    advance=value['advance'], m1_decision=value['m1_decision'],
                    report_sha256=verifier.sha256(path))))

            write()
            with patch.object(verifier, 'CHECKPOINT', str(weight)), \
                    patch.object(verifier, 'CHECKPOINT_SHA', verifier.sha256(weight)), \
                    patch('diagnose_history_stream.CHECKPOINT', str(weight)), \
                    patch('diagnose_history_stream.CHECKPOINT_SHA', verifier.sha256(weight)), \
                    patch.object(verifier, 'load_cache', return_value=(receipt, synthetic_cache)):
                verified = verifier.verify(report_path, receipt_path, config)
                self.assertTrue(verified['terminal'] and verified['accepted'])
                self.assertFalse(verified['ohcas_training_authorized'])
                self.assertFalse(verified['f_training_authorized'])
                original = copy.deepcopy(report)
                report.update(report_for(3, False))
                write()
                retry = verifier.verify(report_path, receipt_path, config)
                self.assertFalse(retry['terminal'] or retry['accepted'])
                earlier_path = root / 'block3/report.json'
                earlier_path.parent.mkdir()
                write(report, earlier_path)
                previous = verifier.check_retry(0, earlier_path, receipt_path, config)
                for advance in (False, True):
                    report.update(report_for(0, advance))
                    report['previous_diagnostic'] = previous
                    write()
                    terminal = verifier.verify(report_path, receipt_path, config)
                    self.assertTrue(terminal['terminal'])
                    self.assertEqual(terminal['accepted'], advance)
                report = original
                write()
                report['seconds'] = 2.
                report_path.write_text(json.dumps(report))
                with self.assertRaises(AssertionError):
                    verifier.verify(report_path, receipt_path, config)
                write()
                config.write_text('changed fixture')
                with self.assertRaises(AssertionError):
                    verifier.verify(report_path, receipt_path, config)
                config.write_text('synthetic fixture')
                cache.write_text('changed fixture')
                # Cache content/provenance validation belongs to the real load_cache.
                with patch.object(verifier, 'load_cache', side_effect=AssertionError('cache changed')):
                    with self.assertRaises(AssertionError):
                        verifier.verify(report_path, receipt_path, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DecisionVerifierTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    report = dict(status='passed', tests=result.testsRun,
                  source_sha256={str(path.resolve()): verifier.sha256(path) for path in
                                 (Path(__file__), Path(verifier.__file__))},
                  real_images_executed=False, real_m1_decision_verified=False,
                  ohcas_training_authorized=False, f_training_authorized=False,
                  scope='Synthetic paired records and mocked checkpoint/cache only; '
                        'does not supply actual diagnostic or provenance evidence.')
    with Path(args.output).open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()

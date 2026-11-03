"""Recompute a completed history diagnostic before consuming its M1 decision."""
import argparse
import json
import math
from pathlib import Path

from diagnose_absent_position import clustered, sha256
from diagnose_history_stream import CHECKPOINT, CHECKPOINT_SHA, check_retry, load_cache
from research.history_stream.diagnostic import (
    MODES, decision, different_sequence_donors, summarize,
)


def check_rows(report, identities):
    """Validate all paired observations before computing the registered gate."""
    assert len(identities) == 128
    donors = different_sequence_donors(identities)
    assert len(report['rows']) == 384
    for index, identity in enumerate(identities):
        for offset, state in enumerate(('11', '01', '10')):
            row = report['rows'][3 * index + offset]
            assert row['identity'] == identity and row['state'] == state
            assert row['wrong_sequence'] == identities[donors[index]]['sequence']
            for mode in MODES:
                assert set(row[mode]) == {'task_loss', 'iou'}
                assert math.isfinite(row[mode]['task_loss']) and row[mode]['task_loss'] >= 0
                assert math.isfinite(row[mode]['iou']) and 0 <= row[mode]['iou'] <= 1
                if state == '11':
                    assert row[mode] == row['zero']
    effects, advance = summarize(report['rows'], clustered)
    assert effects == report['effects'] and advance is report['advance']
    expected = decision(report['injection_layer'], advance)
    assert report['m1_decision'] == expected
    return expected


def verify(report_path, cache_receipt_path, config_path):
    """Use terminal cache provenance and raw pairs, never a status flag alone."""
    report_path = Path(report_path).resolve()
    cache_receipt_path = Path(cache_receipt_path).resolve()
    config_path = Path(config_path).resolve()
    completed_path = report_path.parent / 'completed.json'
    report = json.loads(report_path.read_text())
    completed = json.loads(completed_path.read_text())
    assert report['status'] == completed['status'] == 'complete'
    assert sha256(report_path) == completed['report_sha256']
    assert report['checkpoint'] == CHECKPOINT
    assert report['checkpoint_sha256'] == sha256(CHECKPOINT) == CHECKPOINT_SHA
    assert report['injection_layer'] in (0, 3)
    for key in ('injection_layer', 'advance', 'm1_decision'):
        assert completed[key] == report[key]
    for key in ('complete_bitwise_parity', 'native_zero_bitwise_parity',
                'state_unchanged', 'gradients_absent'):
        assert report[key] is True
    assert math.isfinite(report['seconds']) and report['seconds'] > 0
    assert report['peak_gpu_bytes'] > 0
    sources = report['source_sha256']
    assert sources
    project = Path(__file__).resolve().parents[1]
    for path in (project / 'scripts/diagnose_history_stream.py',
                 project / 'scripts/diagnose_absent_position.py',
                 project / 'research/history_stream/diagnostic.py', config_path):
        assert sources[str(path)] == sha256(path)
    for name, expected in sources.items():
        assert sha256(name) == expected, name
    receipt, cache = load_cache(cache_receipt_path)
    assert report['cache_receipt'] == receipt
    assert report['cache_receipt_sha256'] == sha256(cache_receipt_path)
    identities = [item['identity'] for item in cache]
    assert identities == report['input_manifest']
    result = check_rows(report, identities)
    previous = report['previous_diagnostic']
    earlier = None
    if report['injection_layer'] == 0:
        assert previous is not None
        assert previous == check_retry(
            0, previous['report'], cache_receipt_path, config_path)
        earlier = verify(previous['report'], cache_receipt_path, config_path)
        assert earlier['m1_decision'] == 'retry_embedding'
    else:
        assert previous is None
    evidence = dict(sources)
    for path in (report_path, completed_path, cache_receipt_path,
                 Path(receipt['cache']), Path(receipt['frame_exclusion_audit']),
                 Path(CHECKPOINT), Path(__file__).resolve()):
        evidence[str(path)] = sha256(path)
    for run in receipt['excluded_runs']:
        evidence[run['completed']] = run['completed_sha256']
        evidence[run['manifest']] = run['manifest_sha256']
    evidence.update(receipt['source_sha256'])
    if earlier is not None:
        evidence.update(earlier['evidence_sha256'])
    accepted = result in ('accepted_block3', 'accepted_embedding')
    terminal = accepted or result == 'closed_both_diagnostics_negative'
    return dict(status='passed', m1_decision=result, terminal=terminal,
                accepted=accepted, injection_layer=report['injection_layer'],
                paired_observations=384, samples=128,
                report=str(report_path), report_sha256=sha256(report_path),
                evidence_sha256=evidence, ohcas_training_authorized=False,
                f_training_authorized=False,
                scope='Recomputed saved diagnostic decision and provenance only. '
                      'A retry is nonterminal; an accepted result still needs actual '
                      'OHCAS composition. Not training or an official tracking score.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--cache-receipt', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.report, args.cache_receipt, args.config), indent=2),
          flush=True)


if __name__ == '__main__':
    main()

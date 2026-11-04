"""Read-only M1 decision verification; this never authorizes F training."""
import argparse
import json
from pathlib import Path
import sys

from research.protocol_r.common import ROOT, read, sha


def verify_decision(report_path, cache_receipt, config_path):
    report_path, cache_receipt, config_path = [
        Path(path).resolve() for path in (report_path, cache_receipt, config_path)]
    completed_path = report_path.parent / 'completed.json'
    required = (report_path, completed_path, cache_receipt, config_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(ready=False, stage='WAITING_FOR_M1', missing=missing,
                    f_training_authorized=False)

    # The existing diagnostic imports its sibling script by its standalone name.
    scripts = str(ROOT / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from scripts import diagnose_history_stream as diagnostic
    from research.history_stream.diagnostic import (
        MODES, decision, different_sequence_donors, summarize,
    )

    report, completed = read(report_path), read(completed_path)
    assert report['status'] == completed['status'] == 'complete'
    assert completed['report_sha256'] == sha(report_path)
    assert report['checkpoint'] == diagnostic.CHECKPOINT
    assert report['checkpoint_sha256'] == diagnostic.CHECKPOINT_SHA
    assert sha(report['checkpoint']) == diagnostic.CHECKPOINT_SHA
    receipt, cache = diagnostic.load_cache(cache_receipt)
    assert report['cache_receipt'] == receipt
    assert report['cache_receipt_sha256'] == sha(cache_receipt)
    assert report['source_sha256'][str(config_path)] == sha(config_path)
    for path, expected in report['source_sha256'].items():
        assert sha(path) == expected, path
    assert all(report[key] is True for key in (
        'complete_bitwise_parity', 'native_zero_bitwise_parity',
        'state_unchanged', 'gradients_absent'))

    identities = [item['identity'] for item in cache]
    assert report['input_manifest'] == identities
    assert [(row['identity'], row['state']) for row in report['rows']] == [
        (identity, state) for identity in identities for state in ('11', '01', '10')]
    donors = different_sequence_donors(identities)
    for index, row in enumerate(report['rows']):
        donor = identities[donors[index // 3]]['sequence']
        assert row['wrong_sequence'] == donor
        assert all(mode in row for mode in MODES)
    effects, advance = summarize(report['rows'], diagnostic.clustered)
    assert effects == report['effects'] and report['advance'] is advance
    layer = report['injection_layer']
    assert completed['injection_layer'] == layer
    assert completed['advance'] is advance
    outcome = decision(layer, advance)
    assert report['m1_decision'] == completed['m1_decision'] == outcome
    previous = report['previous_diagnostic']
    checked = diagnostic.check_retry(
        layer, previous['report'] if previous is not None else None,
        cache_receipt, config_path)
    assert checked == previous

    evidence = {str(path): sha(path) for path in required}
    evidence.update(report['source_sha256'])
    evidence[report['checkpoint']] = diagnostic.CHECKPOINT_SHA
    for path in (receipt['cache'], receipt['frame_exclusion_audit']):
        evidence[str(path)] = sha(path)
    evidence.update(receipt['source_sha256'])
    for run in receipt['excluded_runs']:
        evidence[run['completed']] = run['completed_sha256']
        evidence[run['manifest']] = run['manifest_sha256']
    if previous is not None:
        evidence[previous['report']] = previous['report_sha256']
        evidence[previous['completed']] = previous['completed_sha256']
    evidence[str(Path(__file__).resolve())] = sha(__file__)
    return dict(ready=outcome != 'retry_embedding',
        stage='M1_RETRY_REQUIRED' if outcome == 'retry_embedding'
              else 'M1_DECISION_VERIFIED',
        m1_decision=outcome, requires_ohcas_composition=advance,
        f_training_authorized=False, parents_sha256=evidence,
        scope='Saved M1 evidence and recomputed decision only; P selection, '
              'accepted-OHCAS composition, frozen F preparation and five '
              'matched real-image smokes remain caller-owned gates')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--cache-receipt', required=True)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(verify_decision(
        args.report, args.cache_receipt, args.config), indent=2), flush=True)


if __name__ == '__main__':
    main()

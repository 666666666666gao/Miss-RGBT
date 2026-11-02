"""Verify metadata feasibility counts against the native causal sampler."""
import json
from collections import defaultdict
from pathlib import Path
import tempfile
from unittest.mock import patch

import audit_partitioned_fresh_frames as audit
from audit_partitioned_fresh_frames import causal_anchor_count


def main():
    assert causal_anchor_count([1]*20, 20) == 16
    assert causal_anchor_count([1]*19, 19) == 0
    assert causal_anchor_count([1]*10+[0]*10, 20) == 0
    assert causal_anchor_count([1]*11+[0]*9, 20) == 7
    assert causal_anchor_count([1]*12+[0]*8, 20) == 8
    assert causal_anchor_count([0]*19+[1], 20) == 0
    assert causal_anchor_count([1]+[0]*18+[1], 20) == 0
    assert causal_anchor_count([1]*16+[0]*4, 20) == 12
    assert causal_anchor_count([1]*20, 20, 2, 3) == 16
    with tempfile.TemporaryDirectory(prefix='miss_rgbt_verified_audit_') as directory:
        root = Path(directory)
        verification = root/'reports/server28_partitioned_a_20261102_formal/verification_v2.json'
        verification.parent.mkdir(parents=True)
        used, hashes = defaultdict(set), {}
        arms = {}
        for arm in ['utility', 'random']:
            run = root/'runs/server28_partitioned_a_20261102_formal'/arm
            run.mkdir(parents=True)
            completed, manifest = run/'completed.json', run/'input_manifest.jsonl'
            completed.write_text(json.dumps(dict(status='complete')))
            identity = dict(sequence='seq_'+arm, template_frames=[1, 2], search_frames=[3, 4, 5])
            manifest.write_text(json.dumps(dict(identities=[identity]))+'\n')
            arms[arm] = dict(completed_sha256=audit.base.prior.common.sha256(completed),
                input_manifest_sha256=audit.base.prior.common.sha256(manifest))
        verification.write_text(json.dumps(dict(status='passed', phase='formal', arms=arms)))
        with patch.object(audit.base, 'PROJECT', root):
            result = audit.add_verified_partitioned_a_exclusions(verification, used, hashes)
            assert set(used['seq_utility']) == {1, 2, 3, 4, 5}
            assert set(used['seq_random']) == {1, 2, 3, 4, 5}
            assert result['utility_manifest_sha256'] == arms['utility']['input_manifest_sha256']
            saved = json.loads(verification.read_text())
            saved['arms']['random']['input_manifest_sha256'] = '0'*64
            verification.write_text(json.dumps(saved))
            with patch.object(audit.base, 'PROJECT', root):
                try:
                    audit.add_verified_partitioned_a_exclusions(
                        verification, defaultdict(set), {})
                except AssertionError:
                    pass
                else:
                    raise AssertionError('Changed completed manifest was accepted')
            saved['status'] = 'running'
            verification.write_text(json.dumps(saved))
            try:
                audit.add_verified_partitioned_a_exclusions(
                    verification, defaultdict(set), {})
            except AssertionError:
                pass
            else:
                raise AssertionError('Incomplete paired verifier was accepted')

    import hashlib
    source = Path(audit.__file__)
    report = dict(status='PASS', checks=[
        'native anchor lower/upper bounds',
        'strict visible count > 2*(2+3)',
        'minimum sequence length 20',
        'one-direction and zero-visible exclusions',
        'verified completed paired-A manifest acceptance and hash/status rejection',
        'no image/model/GPU access in counting helper'],
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope='Pure metadata helper checks; no dataset read or process launch')
    with tempfile.TemporaryDirectory(prefix='miss_rgbt_frame_audit_') as directory:
        path = Path(directory)/'check.json'
        assert not path.exists()
        path.write_text(json.dumps(report, indent=2)+'\n')
        assert json.loads(path.read_text()) == report
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

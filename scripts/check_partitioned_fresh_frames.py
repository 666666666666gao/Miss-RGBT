"""Verify metadata feasibility counts against the native causal sampler."""
import json
from pathlib import Path
import tempfile

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

    import hashlib
    import audit_partitioned_fresh_frames as audit
    source = Path(audit.__file__)
    report = dict(status='PASS', checks=[
        'native anchor lower/upper bounds',
        'strict visible count > 2*(2+3)',
        'minimum sequence length 20',
        'one-direction and zero-visible exclusions',
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

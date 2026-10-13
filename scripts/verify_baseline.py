import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


original = json.loads(Path('reports/e0/source_sha256.json').read_text())
for path, expected in original.items():
    assert sha(path) == expected, path
    assert sha(Path('/data2/zhouy/CADTrack-main') / path) == expected, path
weight = Path('checkpoints/pretrained/CADTrack_ep0020.pth.tar')
expected = 'd20c8641e4263df3ceadf9e430d0ac96c181c13ad482bca28c2dddef592a8bb8'
assert sha(weight) == expected
report = dict(source_files=len(original), source_exact=True, original_source_exact_after_evaluation=True, checkpoint_sha256=expected,
              checkpoint_bytes=weight.stat().st_size)
Path('reports/baseline_integrity.json').write_text(json.dumps(report, indent=2))
print('BASELINE_INTEGRITY_PASS ' + json.dumps(report), flush=True)

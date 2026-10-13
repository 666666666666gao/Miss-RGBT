import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import rgbt
import timm
import torch
import yaml


torch.manual_seed(2027)
torch.cuda.manual_seed_all(2027)
x = torch.randn(32, 32, device='cuda')
y = x @ x.T
torch.cuda.synchronize()
assert y.shape == (32, 32) and torch.isfinite(y).all()
spec = json.loads(Path('env-spec.json').read_text())
canonical = json.dumps(spec, sort_keys=True, separators=(',', ':'))
report = dict(spec_hash=hashlib.sha256(canonical.encode()).hexdigest()[:8],
              torch=torch.__version__, numpy=np.__version__, timm=timm.__version__,
              cv2=cv2.__version__, yaml=yaml.__version__,
              device=torch.cuda.get_device_name(), shape=list(y.shape),
              value=float(y[0, 0]), finite=True)
assert report['torch'] == '1.12.1+cu113'
assert report['numpy'] == '1.21.5'
assert report['timm'] == '0.9.12'
assert report['cv2'] == '4.8.1'
assert report['yaml'] == '6.0.1'
Path('reports/environment_witness.json').write_text(json.dumps(report, indent=2))
print('WITNESS ' + json.dumps(report), flush=True)

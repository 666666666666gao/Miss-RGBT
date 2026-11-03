"""CPU saved-smoke fixtures, not actual real-image smoke evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from lib.config.cadtrack.config import cfg, update_config_from_file
from research.curriculum_p.check_engine import inputs
from research.curriculum_p.engine import ARMS as CURRICULA, CurriculumState
from research.feedback_f.check_protocol_runner import ProbeModel, optimizer
from research.feedback_f.protocol_runner import step as runner_step
from research.feedback_f.protocol_smokes import verify_smokes
from research.feedback_f.protocol_training import ARMS
from research.protocol_r.common import read, save, sha, tensor_sha


def fixture(root, curriculum='paced+norm'):
    configs = []
    for mode in ARMS:
        config = dict(stage='A', init='ostrack', curriculum_arm=curriculum,
            feedback=dict(mode=mode, width=128), epochs=20, early_stop=False,
            checkpoint_epochs=[10, 16, 17, 18, 19, 20], freeze_all_bn=True,
            repair=False, task_utility=dict(selection='utility'),
            utility_probe_interval=4, utility_weight=.1, patch_mask_probability=.1,
            forward_weight=.1, reverse_weight=0., reverse_margin=.1,
            samples_per_epoch=20000, workers=2,
            warmup_updates=500, official_test_selection_used=False, resume_supported=False,
            microbatch=2, accumulation=2, learning_rate=5e-5, backbone_multiplier=.1,
            mask_seed=65503, development_samples=128, lr_drop_epoch=10, clip_grad_norm=.1,
            output=str(root / mode))
        configs.append(config)
        smoke = Path(config['output'] + '_smoke')
        smoke.mkdir()
        save(smoke / 'run_config.json', config)
        save(smoke / 'initial_state.json', dict(fixture=True, initializer='same'))
        templates, searches, boxes = inputs()
        data = dict(template_images=torch.stack([value[:2] for value in templates]),
                    search_images=torch.stack([value[:2] for value in searches]),
                    search_anno=boxes[:, :2])
        torch.manual_seed(65501)
        model = ProbeModel(mode)
        opt = optimizer(model)
        state = CurriculumState(config['curriculum_arm'], 'cpu', 65502)
        generator = torch.Generator().manual_seed(65503)
        updates = 0
        identities, steps = [], []
        for index in range(8):
            identity = dict(epoch=1, index=index,
                tensors_sha256={key: tensor_sha(value) for key, value in data.items()})
            identities.append(identity)
            row, updates = runner_step(model, opt, config, 1, index,
                                        data, state, generator, updates)
            row['input_sha256'] = identity['tensors_sha256']
            if 'update' in row:
                # CPU fixtures have no GPU allocation; required telemetry is synthetic.
                row.update(peak_gpu_bytes=1, seconds=float(index))
            steps.append(row)
        for name, rows in [('input_manifest.jsonl', identities), ('steps.jsonl', steps)]:
            (smoke / name).write_text(''.join(json.dumps(row) + '\n' for row in rows))
        model = dict(native=torch.ones(1))
        initial = {}
        if mode in ('shallow', 'deep'):
            model['feedback.delta.weight'] = torch.ones(1)
            initial['feedback.delta.weight'] = dict(shape=[1], sha256=tensor_sha(torch.zeros(1)))
        save(smoke / 'feedback_initial_state.json', initial)
        torch.save(dict(config=config, updates=4, epoch=1, resume_supported=False,
            loader_position=dict(epoch=1, consumed_batches=8, batches_per_epoch=10000),
            curriculum=dict(arm=curriculum, microsteps=8), model=model),
            str(smoke / 'smoke.pth.tar'))
        save(smoke / 'smoke_verified.json', dict(status='passed', updates=4,
            fresh_cpu_strict_load=True, all_rounds_reload_bitwise_equal=True,
            checkpoint_sha256=sha(smoke / 'smoke.pth.tar'),
            input_manifest_sha256=sha(smoke / 'input_manifest.jsonl')))
    return configs


class SmokeGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        update_config_from_file('experiments/cadtrack/cadtrack.yaml')

    def test_matched_receipts_and_parent_callbacks(self):
        for curriculum in CURRICULA:
            with self.subTest(curriculum=curriculum), tempfile.TemporaryDirectory() as directory:
                configs = fixture(Path(directory), curriculum)
                checked = []
                result = verify_smokes(configs[0], configs, lambda config: checked.append(config))
                self.assertEqual(checked, configs)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(len(result['evidence_sha256']), 28)
                self.assertTrue(all(sha(path) == digest for path, digest in
                                    result['evidence_sha256'].items()))

    def test_rejects_mismatched_or_incomplete_smokes(self):
        for mutation in ('inputs', 'initial', 'feedback_initial', 'weight',
                         'updates', 'reload', 'config', 'unchanged_feedback',
                         'truncated_steps', 'wrong_update', 'wrong_mode', 'wrong_loss',
                         'mismatched_masks'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                configs = fixture(Path(directory))
                smoke = Path(configs[-1]['output'] + '_smoke')
                receipt = read(smoke / 'smoke_verified.json')
                if mutation == 'inputs':
                    identities = [json.loads(line) for line in
                                  (smoke / 'input_manifest.jsonl').read_text().splitlines()]
                    identities[-1]['tensors_sha256'] = dict(fixture='different')
                    (smoke / 'input_manifest.jsonl').write_text(
                        ''.join(json.dumps(row) + '\n' for row in identities))
                    receipt['input_manifest_sha256'] = sha(smoke / 'input_manifest.jsonl')
                elif mutation == 'initial':
                    save(smoke / 'initial_state.json', dict(fixture='different'))
                elif mutation == 'feedback_initial':
                    initial = read(smoke / 'feedback_initial_state.json')
                    initial['feedback.delta.weight']['sha256'] = 'different'
                    save(smoke / 'feedback_initial_state.json', initial)
                elif mutation == 'weight':
                    with (smoke / 'smoke.pth.tar').open('ab') as stream:
                        stream.write(b'changed')
                elif mutation in ('updates', 'reload'):
                    receipt['updates' if mutation == 'updates' else
                            'all_rounds_reload_bitwise_equal'] = 3 if mutation == 'updates' else False
                elif mutation == 'config':
                    altered = copy.deepcopy(configs[-1])
                    altered['mask_seed'] += 1
                    save(smoke / 'run_config.json', altered)
                elif mutation in ('truncated_steps', 'wrong_update', 'wrong_mode', 'wrong_loss',
                                   'mismatched_masks'):
                    path = smoke / 'steps.jsonl'
                    steps = [json.loads(line) for line in path.read_text().splitlines()]
                    if mutation == 'truncated_steps':
                        steps.pop()
                    elif mutation == 'wrong_update':
                        steps[-1]['update'] = 3
                    elif mutation == 'wrong_mode':
                        steps[-1]['feedback_mode'] = 'single'
                    elif mutation == 'mismatched_masks':
                        steps[-1]['masks_sha256'][-1] = 'changed'
                    else:
                        steps[-1]['loss'] += 1.
                    path.write_text(''.join(json.dumps(row) + '\n' for row in steps))
                else:
                    path = smoke / 'smoke.pth.tar'
                    saved = torch.load(str(path), map_location='cpu')
                    saved['model']['feedback.delta.weight'].zero_()
                    torch.save(saved, str(path))
                    receipt['checkpoint_sha256'] = sha(path)
                save(smoke / 'smoke_verified.json', receipt)
                with self.assertRaises(AssertionError):
                    verify_smokes(configs[0], configs, lambda config: None)


if __name__ == '__main__':
    unittest.main()

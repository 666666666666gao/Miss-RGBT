"""Stdlib control-flow tests with stubbed neural outputs; no Torch/GPU execution."""
import ast
from contextlib import nullcontext
import itertools
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in names]
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


class Tensor:
    def __init__(self, value):
        self.value = value

    def __mul__(self, value):
        value = value.value if isinstance(value, Tensor) else value
        if isinstance(self.value, list):
            return Tensor([item * value for item in self.value])
        return Tensor(self.value * value)

    def __truediv__(self, value):
        return self * (1 / value)

    def view(self, *shape):
        return self

    def mean(self, axis):
        return self

    def max(self):
        return self.value

    def tolist(self):
        return self.value


class Torch:
    float32 = 'float32'
    no_grad = staticmethod(nullcontext)
    cuda = SimpleNamespace(synchronize=lambda: None, current_device=lambda: 0)

    @staticmethod
    def tensor(value, **kwargs):
        return Tensor(value)

    @staticmethod
    def ones(*shape, **kwargs):
        return Tensor([[1, 1]])


class Network:
    def __init__(self):
        self.scores = iter([])
        self.baseline = SimpleNamespace(box_head=SimpleNamespace(
            cal_bbox=lambda *args: Tensor([.55, .55, .1, .1])))

    def __call__(self, *args):
        return [dict(score_map=Tensor(next(self.scores)), size_map=None, offset_map=None)]


class Boxes:
    def __init__(self, n):
        self.rows = [[0., 0., 0., 0.] for _ in range(n)]

    def __getitem__(self, index):
        return self.rows[index]

    def __setitem__(self, index, value):
        self.rows[index] = list(value)


class Numpy:
    float64 = 'float64'

    @staticmethod
    def zeros(shape, **kwargs):
        return Boxes(shape[0])

    @staticmethod
    def isfinite(boxes):
        return SimpleNamespace(all=lambda: all(math.isfinite(x) for row in boxes.rows for x in row))

    @staticmethod
    def savetxt(path, boxes):
        Path(path).write_text('\n'.join(','.join(str(x) for x in row) for row in boxes.rows))


def tracker(policy='current', audit=True, scores=None, interval=5, threshold=.65):
    namespace = load_definitions(ROOT / 'miss_rgbt/tracker.py',
        ['template_update_allowed', 'MissingTracker'], dict(torch=Torch,
        sample_target=lambda image, box, factor, output_sz: ((image.index, tuple(box)), 1., None),
        clip_box=lambda box, *args, **kwargs: box))
    cls = namespace['MissingTracker']
    obj = cls.__new__(cls)
    obj.template_policy, obj.update_audit = policy, audit
    obj.cfg = SimpleNamespace(TEST=SimpleNamespace(TEMPLATE_FACTOR=2, TEMPLATE_SIZE=128,
        SEARCH_FACTOR=4, SEARCH_SIZE=256, UPDATE_INTERVALS=interval, UPDATE_THRESHOLD=threshold))
    obj.num_template = 2
    obj.window = Tensor(1.)
    obj.network = Network()
    obj.network.scores = iter(scores if scores is not None else [.9] * 50)
    obj.preprocessor = SimpleNamespace(process=lambda value: value)
    obj.initialize(SimpleNamespace(index=0, shape=(1000, 1000, 6)), dict(init_bbox=[100, 100, 20, 20]))
    return obj


def image(index):
    return SimpleNamespace(index=index, shape=(1000, 1000, 6))


class TemplateControlsTest(unittest.TestCase):
    def test_current_matches_legacy_rule_exhaustively(self):
        rule = load_definitions(ROOT / 'miss_rgbt/tracker.py', ['template_update_allowed'], {})['template_update_allowed']
        for frame, interval, score, threshold, available in itertools.product(
                range(1, 31), [5, 10], [.0, .6, .65, .6500000000000001, .7, 1.],
                [.6, .65, .7], [[1, 1], [1, 0], [0, 1]]):
            expected = frame % interval == 0 and score > threshold
            self.assertEqual(rule('current', frame, interval, score, threshold, available), expected)
            self.assertFalse(rule('off', frame, interval, score, threshold, available))
            self.assertEqual(rule('complete_only', frame, interval, score, threshold, available),
                             expected and available == [1, 1])

    def test_off_keeps_two_slots_but_still_commits_position(self):
        obj = tracker('off')
        initial_templates, initial_state = list(obj.templates), list(obj.state)
        for i in range(1, 11):
            obj.track(image(i), [1, 1])
            self.assertFalse(obj.last_update_event['template_written'])
        self.assertEqual(obj.templates, initial_templates)
        self.assertEqual(len(obj.templates), 2)
        self.assertNotEqual(obj.state, initial_state)

    def test_current_writes_missing_availability_and_preserves_initial(self):
        obj = tracker()
        initial = obj.templates[0]
        for i in range(1, 6):
            obj.track(image(i), [0, 1])
        self.assertTrue(obj.last_update_event['template_written'])
        self.assertEqual(obj.templates[0], initial)
        self.assertEqual(obj.templates[1][0], 5)
        self.assertEqual(obj.availability[1].tolist(), [[0, 1]])
        self.assertEqual(obj.last_template_available, [0, 1])
        self.assertEqual(len(obj.templates), 2)

    def test_complete_only_skips_partial_without_delayed_write(self):
        obj = tracker('complete_only')
        for i in range(1, 6):
            obj.track(image(i), [1, 0])
        self.assertEqual(obj.last_template_write, 0)
        obj.track(image(6), [1, 1])
        self.assertEqual(obj.last_template_write, 0)
        for i in range(7, 11):
            obj.track(image(i), [1, 1])
        self.assertEqual(obj.last_template_write, 10)
        self.assertEqual(obj.last_update_event['template_age_before'], 10)

    def test_complete_only_equals_current_for_complete_stream(self):
        current, complete = tracker('current'), tracker('complete_only')
        for i in range(1, 21):
            self.assertEqual(current.track(image(i), [1, 1]), complete.track(image(i), [1, 1]))
            self.assertEqual(current.templates, complete.templates)
            self.assertEqual(current.last_update_event, complete.last_update_event)

    def test_audit_does_not_change_stubbed_tracking(self):
        observed, plain = tracker(audit=True), tracker(audit=False)
        for i in range(1, 16):
            available = [1, i % 2]
            self.assertEqual(observed.track(image(i), available), plain.track(image(i), available))
            self.assertEqual(observed.templates, plain.templates)

    def test_threshold_equality_is_not_a_write(self):
        obj = tracker(scores=[.65] * 5)
        for i in range(1, 6):
            obj.track(image(i), [1, 1])
        self.assertEqual(obj.last_template_write, 0)

    def test_interval_override_and_sequence_reset(self):
        obj = tracker(interval=10)
        for i in range(1, 11):
            obj.track(image(i), [1, 1])
            self.assertEqual(obj.last_template_write, 10 if i == 10 else 0)
        obj.initialize(image(0), dict(init_bbox=[100, 100, 20, 20]))
        self.assertEqual((obj.frame_id, obj.last_template_write, obj.last_template_available), (0, 0, [1, 1]))
        self.assertEqual(obj.templates[0], obj.templates[1])

    def test_actual_sequence_control_flow_holds_without_advancing_update_clock(self):
        obj = tracker()
        mask = [[1, 1], [1, 1], [0, 0], [0, 1], [1, 1], [0, 0], [1, 0], [1, 1]]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['bbox', 'receipts', 'template_updates']:
                (root / name / 'dataset').mkdir(parents=True)
            namespace = load_definitions(ROOT / 'scripts/evaluate_model.py', ['sequence'], dict(
                TRACKER=obj, READ_FRAME=lambda r, t, dtype: image(int(r)), TORCH=Torch,
                ROOT=root, PHYSICAL_GPUS=[2], UPDATE_AUDIT=True, np=Numpy,
                time=SimpleNamespace(time=lambda: 0.), json=json,
                print=lambda *args, **kwargs: None,
                sha=lambda path: __import__('hashlib').sha256(Path(path).read_bytes()).hexdigest(),
                save=lambda path, value: Path(path).write_text(json.dumps(value)),
            ))
            receipt = namespace['sequence'](dict(dataset='dataset', sequence='seq', frames=len(mask),
                init_bbox=[100, 100, 20, 20], rgb=[str(i) for i in range(len(mask))],
                tir=[str(i) for i in range(len(mask))], mask=mask))
            events = json.loads((root / 'template_updates/dataset/seq.json').read_text())['events']
            self.assertEqual(len(events), 7)
            self.assertEqual([event['track_call'] for event in events], [1, 1, 2, 3, 3, 4, 5])
            self.assertEqual([event['video_frame'] for event in events if event['template_written']], [7])
            self.assertEqual(receipt['template_writes'], 1)
            self.assertEqual(events[-1]['video_template_age_before'], 7)
            self.assertEqual(events[-1]['template_age_before'], 5)
            rows = (root / 'bbox/dataset/seq.txt').read_text().splitlines()
            self.assertEqual(rows[1], rows[2])
            self.assertEqual(rows[4], rows[5])


if __name__ == '__main__':
    unittest.main()

"""Check F final-five job plans without loading test data or launching jobs."""
import argparse
import copy
import json
from pathlib import Path
import unittest

import official
from prepare import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    test = unittest.TestCase()
    paths = []
    for curriculum in official.trainer.P_ARMS:
        for arm in official.trainer.ARMS:
            rows = [dict(epoch=e, path='/synthetic/'+arm+'/epoch%02d' % e,
                         sha256=str(e)*4) for e in range(16, 21)]
            completed = dict(status='complete', epochs=20, feedback=dict(mode=arm),
                curriculum_arm=curriculum, official_test_selection_used=False,
                all_checkpoints_retained=True, final_five=rows, selected_epoch=18,
                selected_checkpoint=rows[2]['path'], selected_sha256=rows[2]['sha256'])
            jobs = official.jobs_for(curriculum, arm, completed)
            assert len(jobs) == 10
            assert [(job['epoch'], job['condition']) for job in jobs] == [
                (e, condition) for e in range(16, 21) for condition in ['complete', 'miss']]
            selected = [job for job in jobs if job['development_selected']]
            assert len(selected) == 2 and selected[0]['checkpoint'] == selected[1]['checkpoint']
            for job in jobs:
                assert job['expected_sequences'] == 479 and job['expected_frames'] == 337353
                command = official.command_for(job, 0)
                assert '--limit' not in command and command[-2:] == ['--workers-per-gpu', '4']
                paths.append(job['output'])
            for key, value in [('epochs', 19), ('official_test_selection_used', True),
                               ('selected_sha256', 'wrong')]:
                changed = dict(completed, **{key: value})
                with test.assertRaises(AssertionError):
                    official.jobs_for(curriculum, arm, changed)
            changed = copy.deepcopy(completed)
            changed['final_five'].pop()
            with test.assertRaises(AssertionError):
                official.jobs_for(curriculum, arm, changed)
    assert len(paths) == len(set(paths)) == 160
    report = dict(status='PASS', unique_jobs=160, jobs_per_arm=10,
        selected_checkpoint_paired=True, full_inventory_only=True,
        source_sha256={str(path.resolve()): sha(path)
                       for path in [Path(__file__), Path(official.__file__)]},
        scope='Synthetic job-plan checks only; no campaign, processes, data or official metrics')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

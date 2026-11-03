"""Compare history feasibility against exhaustive small synthetic sequences."""
import argparse
import itertools
import json
from pathlib import Path

from research.history_stream.metadata import history_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cases = 0
    for length in range(1, 9):
        for visible in itertools.product((False, True), repeat=length):
            for excluded in ([], list(range(0, length, 3))):
                for age in (1, 3, 64):
                    fresh = [i for i in range(length)
                             if visible[i] and i not in excluded]
                    pairs = [(h, s) for s in fresh for h in fresh
                             if 1 <= s - h <= age and s - 1 in fresh
                             and len([t for t in fresh if t < h]) >= 2]
                    result = history_candidates(visible, excluded, age)
                    assert result['history_search_pairs'] == len(pairs)
                    assert result['eligible_searches'] == len({s for h, s in pairs})
                    assert result['fresh_visible_frames'] == len(fresh)
                    candidate = result['first_candidate']
                    assert (candidate is None) == (not pairs)
                    if candidate:
                        assert (candidate['history_frame'], candidate['search_frame']) in pairs
                        assert len(set(candidate['template_frames'])) == 2
                        assert max(candidate['history_template_frames']) < candidate['history_frame']
                        assert candidate['crop_reference_frame'] in fresh
                    cases += 1
    with Path(args.output).open('x') as stream:
        json.dump(dict(status='passed', cases=cases, images_opened=0,
            scope='Exhaustive synthetic metadata arithmetic; not a data gate.'), stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='passed', cases=cases)), flush=True)


if __name__ == '__main__':
    main()

"""Count causal training anchors remaining after frozen historical exclusions."""
import argparse
import json
from pathlib import Path

from lib.train.dataset import LasHeR
import train_partitioned_a as base


def causal_anchor_count(visible, sequence_length, num_template_frames=2,
                        num_search_frames=3):
    """Count anchors reachable by the native causal sampler as its gap grows."""
    if sequence_length < 20 or sum(visible) <= 2*(num_template_frames+num_search_frames):
        return 0
    return sum(bool(visible[anchor])
        and any(visible[:anchor])
        and sum(visible[anchor+1:]) >= num_search_frames
        for anchor in range(num_template_frames-1, sequence_length-num_search_frames))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    names_path = Path('lib/train/data_specs/lasher_train.txt')
    names = names_path.read_text().splitlines()
    assert len(names) == len(set(names)) == 882
    names = [name for index, name in enumerate(names) if index % 5]
    used, hashes = base.prior.exclusions()
    dense = base.PROJECT/'reports/server28_dense_geometry_20261102/report.json'
    hashes[str(dense)] = base.prior.common.sha256(dense)
    for manifest in json.loads(dense.read_text())['cache_manifests'].values():
        for entry in manifest['input_manifest']:
            identity = entry['identity']
            used[identity['sequence']].update(
                identity['template_frames']+identity['search_frames'])
    run_config = json.loads((
        base.PROJECT/'reports/server28_partitioned_a_20261102_formal/utility.json'
    ).read_text())
    dataset = LasHeR(root=run_config['dataset'], split='train', dtype='rgbrgb')
    dataset.sequence_list = names
    results = []
    for index, name in enumerate(names):
        info = dataset.get_sequence_info(index)
        visible = (info['visible'].bool() & info['valid'].bool()).tolist()
        excluded = sorted(used[name])
        assert not excluded or excluded[-1] < len(visible), name
        for frame in excluded:
            visible[frame] = False
        anchors = causal_anchor_count(visible, len(visible))
        results.append(dict(sequence=name, frames=len(visible), visible=sum(visible),
            excluded=len(excluded), causal_anchors=anchors))
    result = dict(status='PASS', sequences=len(results), total_fresh_anchors=sum(
        row['causal_anchors'] for row in results),
        zero_anchor_sequences=sum(row['causal_anchors'] == 0 for row in results),
        minimum_anchors=min(row['causal_anchors'] for row in results),
        median_anchors=sorted(row['causal_anchors'] for row in results)[len(results)//2],
        required_templates=2, required_searches=3,
        minimum_visible_for_sampler=2*(2+3)+1, minimum_sequence_length=20,
        excluded_manifest_sha256=hashes,
        excluded_scope='Completed historical diagnostics/crops via exclusions() and '
                       'completed dense-geometry cache manifests only; excludes no '
                       'active or growing partitioned-A/P manifests',
        source_sha256={str(path.resolve()): base.prior.common.sha256(path) for path in [
            Path(__file__), Path(base.__file__),
            Path('lib/train/data/sampler.py'), Path('miss_rgbt/data.py'),
            Path('lib/train/dataset/lasher.py')]},
        sequence_details=results,
        scope='Metadata-only sampler feasibility; does not open images or estimate '
              'augmentation crop acceptance')
    base.save(output, result)
    print(json.dumps({key: value for key, value in result.items()
                      if key != 'sequence_details'}), flush=True)


if __name__ == '__main__':
    main()

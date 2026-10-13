"""Render saved real-GT paired diagnostics, without model execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    path = Path(args.report)
    results = json.loads(path.read_text())['results']
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8))
    labels = ['11_before_first_missing', '11_return_1', '11_return_2_5', '11_return_6_20', '11_return_21_plus']
    for row, (name, result) in enumerate(results.items()):
        left, right = axes[row]
        states = [result['states'][state] for state in ['01', '10', '11', '00']]
        x = np.arange(4)
        left.bar(x - .18, [state['complete_precision20_pct'] for state in states], .36,
                 color='#365b7c', label='Complete-input baseline')
        left.bar(x + .18, [state['missing_precision20_pct'] for state in states], .36,
                 color='#cc8249', label='Missing-input baseline')
        left.set_xticks(x, ['TIR only', 'RGB only', 'Both present', 'Neither'])
        left.set_ylim(0, 105)
        left.set_ylabel('Paired frame precision at 20 pixels (%)')
        left.set_title(name + ': same frames, same checkpoint')
        histories = [result['histories'][label] for label in labels]
        values = [history['delta_mean_iou_pp'] for history in histories]
        right.bar(np.arange(5), values, color='#79628a', width=.65)
        right.axhline(0, color='#777777', lw=.8)
        right.set_xticks(np.arange(5), ['Before\nmissing', '1', '2-5', '6-20', '21+'])
        right.set_xlabel('Consecutive complete-input frames since last missing frame')
        right.set_ylabel('Missing minus complete mean IoU (pp)')
        right.set_ylim(-18, 2)
        right.set_title(name + ': both sensors currently present')
        for i, (value, history) in enumerate(zip(values, histories)):
            right.text(i, value - .9, '{:.2f}\nn={}'.format(value, history['frames']),
                       ha='center', va='top', fontsize=8)
        for axis in [left, right]:
            axis.spines[['top', 'right']].set_visible(False)
            axis.grid(axis='y', alpha=.18)
            axis.set_axisbelow(True)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=9, loc='lower center', bbox_to_anchor=(.5, .055), ncol=2)
    fig.suptitle('Missing-input damage can persist after sensors return', fontsize=15, y=.98)
    fig.text(.5, .015, 'Descriptive paired E0 diagnostics, not official PR/SR. First frames and invalid GT excluded.\n'
             'Duration groups contain different frames; this plot does not establish an increasing causal trend.',
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=[0, .075, 1, .955], h_pad=2)
    for suffix in ['svg', 'png']:
        fig.savefig(str(path.parent / ('paired_baseline.' + suffix)), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()

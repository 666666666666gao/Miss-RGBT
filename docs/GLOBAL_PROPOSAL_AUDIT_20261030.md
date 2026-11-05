# Global and Coarse-Tile Proposal Audit

## Protocol

Historical A utility epoch20; frozen native tracker and initial templates.
CPU-only audit on first8 fixed val97 sequences. For each observed11/01/10
state, sample25% and75% positions of its observed-frame list, deduplicated:
46 observations,45 with valid positive-size GT. No GT or tracker score
selects proposals or sample frames. Every point evaluates11 proposals:
native previous-box crop, image-sized whole-frame square and nine
half-image-side crops at quarter/half/three-quarter centers. All resize to
the native search size. Both native Hann-window and raw responses are
evaluated. Ground truth is used only afterward for geometric crop support
and predicted-box IoU.

This replays candidates at prerecorded GPU trajectory boxes. It is not
closed-loop recovery, a threshold calibration, full val97 tracking or an
official benchmark.

## Results

| Rule | Mean IoU | IoU >= 0.5 | Changes >0.1 IoU vs native |
|---|---:|---:|---:|
| Native crop + Hann | 0.56452 | 32/45 | Reference |
| Best candidate by GT Hann IoU (oracle) | 0.60123 | 32/45 | 6 improved |
| Select by Hann peak | 0.54660 | 30/45 | 2 harmed, 2 improved |
| Select by raw peak, no Hann | 0.42242 | 23/45 | 12 harmed, 2 improved |

Native crop lacked GT crop support at12 observations. Whole-frame/coarse
candidates restored geometric crop support in all12, but even the best
GT-IoU candidate did not achieve IoU0.5 at any of those12 points.
Therefore coverage is necessary but insufficient; candidate identity and
localization remain unsolved. Response-based selection does not recover
the oracle benefit, and dropping Hann degrades substantially.

The dominant severe-drift sequence `leftdrillmasterstanding` had six
unsupported sampled points. Proposal coverage was restored at all six,
yet oracle mean IoU was only0.09669. In `ab_pingpongball`, only five of six
points had valid-size GT; all methods remained near zero IoU, with no
reliable recovery. Hann selection improves state11 mean IoU but worsens
both single-modality states; raw response has12 large harms. No intervention is
promoted.

This rejects a simplistic full-frame or coarse-grid recovery claim. Next
mechanistic work must discriminate target identity from background and
prove closed-loop gains with fixed train-derived rules and native full
trajectories before spending GPU training.

## Evidence

- `reports/server28-global-proposals-20261030/launch.json`
- `reports/server28-global-proposals-20261030/report.json`
- `reports/server28-global-proposals-20261030/analysis.json`
- `reports/server28-global-proposals-20261030/completed.json`
- CPU worker finished normally in343.02s,46 observations/506 proposals.
- Center batched predictions were compared against native single-crop
  tracking at every point; the worker asserts coordinate equivalence.
- Frozen runtime and all four concurrent GPU trainings were unchanged.
- At16:48Z, the operator's GPU cap command had taken effect on all four
  devices: 250W limits and temperatures84/81/80/75C.
